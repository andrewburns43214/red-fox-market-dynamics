"""Quota-aware PropLine collection, caching, projection publication, and ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import requests

from prop_projection import normalized_name, observation_hash, parse_time, project_event, public_projection, utc_now
from prop_projection_config import API_BASE, LOCAL_DAILY_REQUEST_CAP, REQUEST_TIMEOUT_SECONDS, SPORTS


DATA_ROOT = Path(os.environ.get("REDFOX_PROP_DATA_DIR", "data/prop_projection"))
PUBLIC_PATH = Path(os.environ.get("REDFOX_PROP_PUBLIC_PATH", "data/prop_projections.json"))


def _read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def process_lock(path=DATA_ROOT / "collector.lock"):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                yield False
                return
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
                return
        yield True
    finally:
        try:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


class RequestBudget:
    def __init__(self, path=DATA_ROOT / "quota.json", cap=LOCAL_DAILY_REQUEST_CAP):
        self.path, self.cap = path, cap
        self.day = utc_now().date().isoformat()
        self.state = _read_json(path, {})
        if self.state.get("day") != self.day:
            self.state = {"day": self.day, "local_used": 0}

    def reserve(self):
        if int(self.state.get("local_used", 0)) >= self.cap:
            raise RuntimeError("local_daily_request_cap_reached")
        self.state["local_used"] = int(self.state.get("local_used", 0)) + 1
        _atomic_json(self.path, self.state)

    def update_headers(self, headers):
        for source, target in (("X-Daily-Limit", "provider_limit"), ("X-Daily-Used", "provider_used"), ("X-Daily-Remaining", "provider_remaining"), ("X-Daily-Reset", "provider_reset")):
            if headers.get(source) is not None:
                self.state[target] = headers[source]
        _atomic_json(self.path, self.state)


class PropLineClient:
    def __init__(self, api_key=None, session=None, budget=None):
        self.api_key = api_key or os.environ.get("PROPLINE_API_KEY", "")
        self.session = session or requests.Session()
        self.budget = budget or RequestBudget()

    def get(self, path, params=None):
        if not self.api_key:
            raise RuntimeError("PROPLINE_API_KEY_not_configured")
        self.budget.reserve()
        response = self.session.get(API_BASE + path, params=params or {}, headers={"X-API-Key": self.api_key}, timeout=REQUEST_TIMEOUT_SECONDS)
        self.budget.update_headers(response.headers)
        response.raise_for_status()
        return response.json()

    def bulk_odds(self, provider_sport, markets):
        return self.get(f"/sports/{provider_sport}/odds", {"markets": ",".join(markets)})

    def context(self, provider_sport, event_id):
        return self.get(f"/sports/{provider_sport}/events/{event_id}/context")

    def scores(self, provider_sport):
        return self.get(f"/sports/{provider_sport}/scores")


class RosterResolver:
    """Official-roster identity barrier. Failure rejects players; it never guesses."""
    def __init__(self, session=None, root=DATA_ROOT / "rosters"):
        self.session, self.root = session or requests.Session(), root

    def _cache(self, key, loader, max_age=86400):
        path = self.root / f"{re.sub(r'[^a-zA-Z0-9_.-]', '_', key)}.json"
        if path.exists() and time.time() - path.stat().st_mtime < max_age:
            return _read_json(path, {})
        value = loader()
        _atomic_json(path, value)
        return value

    def _espn_team_index(self, sport):
        league = "nfl" if sport == "nfl" else "college-football"
        def load():
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/{league}/teams"
            payload = self.session.get(url, params={"limit": 1000}, timeout=12).json()
            teams = payload.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
            return {str(item.get("team", {}).get("id")): item.get("team", {}) for item in teams}
        return self._cache(f"{sport}_team_index", load)

    def _espn_roster(self, sport, team_name):
        index = self._espn_team_index(sport)
        wanted = normalized_name(team_name)
        matches = []
        for team_id, team in index.items():
            names = [team.get(k) for k in ("displayName", "shortDisplayName", "name", "location", "slug", "abbreviation")]
            if wanted in {normalized_name(x) for x in names if x} or any(normalized_name(x) and normalized_name(x) in wanted for x in names if x):
                matches.append((team_id, team))
        if len(matches) != 1:
            return [], []
        team_id, team = matches[0]
        league = "nfl" if sport == "nfl" else "college-football"
        def load():
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/{league}/teams/{team_id}/roster"
            return self.session.get(url, timeout=12).json()
        payload = self._cache(f"{sport}_{team_id}_roster", load, max_age=21600)
        athletes = []
        for group in payload.get("athletes", []):
            for athlete in group.get("items", []) if isinstance(group, dict) else []:
                athletes.append(athlete.get("fullName") or athlete.get("displayName"))
        tokens = [team.get("abbreviation"), team.get("shortDisplayName")]
        return [x for x in athletes if x], [x for x in tokens if x]

    def _mlb_roster(self, team_id):
        numeric = str(team_id or "").split(":")[-1]
        if not numeric.isdigit():
            return []
        def load():
            url = f"https://statsapi.mlb.com/api/v1/teams/{numeric}/roster"
            return self.session.get(url, params={"rosterType": "active"}, timeout=12).json()
        payload = self._cache(f"mlb_{numeric}_roster", load, max_age=21600)
        return [item.get("person", {}).get("fullName") for item in payload.get("roster", []) if item.get("person", {}).get("fullName")]

    def for_event(self, sport, event):
        output = {}
        for side in ("away", "home"):
            team = str(event.get(f"{side}_team") or "")
            if sport == "mlb":
                names, tokens = self._mlb_roster(event.get(f"{side}_team_id")), []
            else:
                names, tokens = self._espn_roster(sport, team)
            output[team], output[f"{team}__tokens"] = names, tokens
        return output

    def prefetch(self, sport, events, workers=8):
        """Bounded parallel official-roster fetch; returns event-id keyed maps."""
        events = [event for event in events if event.get("bookmakers")]
        if not events:
            return {}
        if sport in {"nfl", "ncaaf"}:
            self._espn_team_index(sport)  # one cached league index before threads
        output = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.for_event, sport, event): str(event.get("id") or "") for event in events}
            for future in as_completed(futures):
                try:
                    output[futures[future]] = future.result()
                except Exception:
                    output[futures[future]] = {}
        return output


def _events(payload):
    if isinstance(payload, list):
        return payload
    for key in ("events", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def _poll_interval_seconds(events, now):
    leads = []
    for event in events:
        start = parse_time(event.get("commence_time"))
        if start and start > now:
            leads.append((start - now).total_seconds())
    if not leads:
        return 3600
    lead = min(leads)
    if lead <= 3600:
        return 12 * 60
    if lead <= 6 * 3600:
        return 30 * 60
    return 60 * 60


def _append_changed(path, payload, digest, prior_digest):
    if digest == prior_digest:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")) + "\n")
    return True


def _event_change_summary(event, sport, digest, captured_at):
    markets = 0
    outcomes = 0
    for book in event.get("bookmakers") or []:
        for market in book.get("markets") or []:
            markets += 1
            outcomes += len(market.get("outcomes") or [])
    return {
        "captured_at": captured_at.isoformat(), "sport": sport, "hash": digest,
        "event_id": str(event.get("id") or ""), "commence_time": event.get("commence_time"),
        "away_team": event.get("away_team"), "home_team": event.get("home_team"),
        "book_count": len(event.get("bookmakers") or []), "market_blocks": markets,
        "outcome_count": outcomes,
    }


def _retain_final_pregame(state, current, now):
    published = state.setdefault("published", {})
    for item in current:
        published[f"{item.get('sport')}:{item.get('event_id')}"] = item
    keep = []
    active_keys = {f"{item.get('sport')}:{item.get('event_id')}" for item in current}
    for key, item in list(published.items()):
        start = parse_time(item.get("commence_time"))
        if not start or (now - start).total_seconds() > 8 * 3600:
            published.pop(key, None)
            continue
        if key not in active_keys and start <= now and item.get("status") == "AVAILABLE":
            keep.append({**item, "final_pregame": True})
    return current + keep


def run_collection(client=None, resolver=None, force=False, now=None):
    now, client, resolver = now or utc_now(), client or PropLineClient(), resolver or RosterResolver()
    state_path = DATA_ROOT / "state.json"
    state = _read_json(state_path, {"sports": {}, "event_hashes": {}, "event_seen": {}, "contexts": {}})
    projections = []
    for sport, config in SPORTS.items():
        if not config["enabled"]:
            continue
        sport_state = state["sports"].get(sport, {})
        cached_path = DATA_ROOT / "cache" / f"{sport}.json"
        cached = _read_json(cached_path, [])
        interval = _poll_interval_seconds(_events(cached), now)
        last_poll = parse_time(sport_state.get("last_poll"))
        due = force or not last_poll or (now - last_poll).total_seconds() >= interval
        payload = cached
        if due:
            try:
                payload = client.bulk_odds(config["provider_key"], config["markets"])
                _atomic_json(cached_path, payload)
                state["sports"][sport] = {"last_poll": now.isoformat(), "last_success": now.isoformat()}
            except Exception as error:
                state["sports"][sport] = {**sport_state, "last_error": type(error).__name__, "last_error_at": now.isoformat()}
        upcoming_events = [event for event in _events(payload) if (parse_time(event.get("commence_time")) or now) > now]
        rosters_by_event = resolver.prefetch(sport, upcoming_events)
        for event in upcoming_events:
            start = parse_time(event.get("commence_time"))
            if not start or start <= now:
                continue
            digest = observation_hash(event)
            event_key = f"{sport}:{event.get('id')}"
            _append_changed(DATA_ROOT / "observations.jsonl", _event_change_summary(event, sport, digest, now), digest, state["event_hashes"].get(event_key))
            state["event_hashes"][event_key] = digest
            state.setdefault("event_seen", {})[event_key] = now.isoformat()
            context = state.get("contexts", {}).get(event_key, {})
            # MLB context is refreshed at most once per six hours and once in
            # the final hour, which keeps a full slate economical.
            if sport == "mlb":
                context_at = parse_time(context.get("_fetched_at"))
                lead = (start - now).total_seconds()
                context_due = not context_at or (now - context_at).total_seconds() >= (1800 if lead <= 3600 else 21600)
                if due and context_due:
                    try:
                        context = client.context(config["provider_key"], event.get("id"))
                        context["_fetched_at"] = now.isoformat()
                        state.setdefault("contexts", {})[event_key] = context
                    except Exception:
                        pass
            try:
                rosters = rosters_by_event.get(str(event.get("id") or ""), {})
                projection = project_event(sport, event, rosters, context=context, now=now)
            except Exception as error:
                projection = {
                    "sport": sport, "event_id": str(event.get("id") or ""), "away_team": event.get("away_team", ""),
                    "home_team": event.get("home_team", ""), "commence_time": event.get("commence_time"),
                    "model_version": config["model_version"], "provider": "PropLine", "generated_at": now.isoformat(),
                    "status": "UNAVAILABLE", "confidence": "INSUFFICIENT", "reason": f"validation_error:{type(error).__name__}",
                }
            private_lines = projection.pop("_private_lines", [])
            audit = {"projection": projection, "canonical_lines": private_lines}
            hashable_projection = {key: value for key, value in projection.items() if key not in {"generated_at", "oldest_observation_age_minutes"}}
            projection_digest = hashlib.sha256(json.dumps({"projection": hashable_projection, "canonical_lines": private_lines}, sort_keys=True, default=str).encode()).hexdigest()
            _append_changed(DATA_ROOT / "projection_ledger.jsonl", audit, projection_digest, state.get("projection_hashes", {}).get(event_key))
            state.setdefault("projection_hashes", {})[event_key] = projection_digest
            projections.append(public_projection(projection))
    # Keep hash/context state bounded; long-run model evaluation lives in the
    # compact projection/resolution ledgers, not in an ever-growing raw cache.
    cutoff = now.timestamp() - 14 * 86400
    for event_key, seen in list(state.get("event_seen", {}).items()):
        parsed = parse_time(seen)
        if not parsed or parsed.timestamp() < cutoff:
            state["event_seen"].pop(event_key, None)
            state.get("event_hashes", {}).pop(event_key, None)
            state.get("contexts", {}).pop(event_key, None)
            state.get("projection_hashes", {}).pop(event_key, None)
    projections = _retain_final_pregame(state, projections, now)
    payload = {"schema_version": 1, "generated_at": now.isoformat(), "projections": projections}
    _atomic_json(PUBLIC_PATH, payload)
    _atomic_json(state_path, state)
    return payload


def _jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def run_resolution(client=None, now=None, force=False):
    """Privately join frozen pregame projections to final team scores."""
    now, client = now or utc_now(), client or PropLineClient()
    ledger = _jsonl(DATA_ROOT / "projection_ledger.jsonl")
    latest = {}
    for entry in ledger:
        projection = entry.get("projection", {})
        if projection.get("status") == "AVAILABLE":
            latest[f"{projection.get('sport')}:{projection.get('event_id')}"] = projection
    state_path = DATA_ROOT / "resolution_state.json"
    state = _read_json(state_path, {"sports": {}, "resolved": {}})
    for sport, config in SPORTS.items():
        if not config["enabled"]:
            continue
        candidates = {key: value for key, value in latest.items() if value.get("sport") == sport and (parse_time(value.get("commence_time")) or now) < now and key not in state["resolved"]}
        if not candidates:
            continue
        last = parse_time(state["sports"].get(sport))
        if not force and last and (now - last).total_seconds() < 6 * 3600:
            continue
        try:
            scores = _events(client.scores(config["provider_key"]))
        except Exception:
            continue
        state["sports"][sport] = now.isoformat()
        for score in scores:
            key = f"{sport}:{score.get('id') or score.get('event_id')}"
            if key not in candidates or str(score.get("status") or "").lower() not in {"final", "completed", "complete"}:
                continue
            projection = candidates[key]
            try:
                actual_away, actual_home = int(score["away_score"]), int(score["home_score"])
            except (KeyError, TypeError, ValueError):
                continue
            state["resolved"][key] = {
                "sport": sport, "event_id": projection["event_id"], "model_version": projection["model_version"],
                "confidence": projection["confidence"], "projected_away": projection["away_score"],
                "projected_home": projection["home_score"], "actual_away": actual_away, "actual_home": actual_home,
                "away_error": projection["away_score"] - actual_away, "home_error": projection["home_score"] - actual_home,
                "resolved_at": now.isoformat(),
            }
    rows = list(state["resolved"].values())
    metrics = {}
    for sport in SPORTS:
        errors = [error for row in rows if row["sport"] == sport for error in (row["away_error"], row["home_error"])]
        if errors:
            metrics[sport] = {"team_scores": len(errors), "mae": round(sum(abs(x) for x in errors) / len(errors), 3), "bias": round(sum(errors) / len(errors), 3)}
    _atomic_json(state_path, state)
    _atomic_json(DATA_ROOT / "performance.json", {"generated_at": now.isoformat(), "metrics": metrics, "games": rows})
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Red Fox isolated PropLine projection collector")
    parser.add_argument("command", choices=("collect", "resolve", "status"), nargs="?", default="collect")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "status":
        state = _read_json(DATA_ROOT / "state.json", {})
        public = _read_json(PUBLIC_PATH, {"projections": []})
        print(json.dumps({"configured": bool(os.environ.get("PROPLINE_API_KEY")), "state": state.get("sports", {}), "projection_count": len(public.get("projections", []))}, indent=2))
        return 0
    if args.command == "resolve":
        with process_lock() as acquired:
            if not acquired:
                return 0
            try:
                print(json.dumps(run_resolution(force=args.force), indent=2))
            except Exception as error:
                print(f"[props] resolution unavailable: {type(error).__name__}", file=sys.stderr)
        return 0
    with process_lock() as acquired:
        if not acquired:
            print("[props] collector already running; skipped")
            return 0
        try:
            payload = run_collection(force=args.force)
            try:
                run_resolution(force=False)
            except Exception:
                pass
            available = sum(item.get("status") == "AVAILABLE" for item in payload["projections"])
            print(f"[props] published {available}/{len(payload['projections'])} available projections")
            return 0
        except Exception as error:
            # The existing RF runner must never fail because props are absent.
            print(f"[props] unavailable: {type(error).__name__}", file=sys.stderr)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
