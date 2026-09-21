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
from zoneinfo import ZoneInfo

import requests

from prop_projection import normalized_name, observation_hash, parse_time, project_event, public_projection, utc_now
from prop_projection_config import API_BASE, LOCAL_DAILY_REQUEST_CAP, REQUEST_TIMEOUT_SECONDS, SPORTS
from prop_projection_v2 import project_event_v2
from prop_projection_prospective import freeze_candidate, frozen_candidates, grade_candidate, performance_summary


DATA_ROOT = Path(os.environ.get("REDFOX_PROP_DATA_DIR", "data/prop_projection"))
PUBLIC_PATH = Path(os.environ.get("REDFOX_PROP_PUBLIC_PATH", "data/prop_projections.json"))


def _read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _atomic_json(path, payload, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    if mode is not None:
        os.chmod(temporary, mode)
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


def espn_team_matches(provider_name, team):
    names = [team.get(k) for k in ("displayName", "shortDisplayName", "name", "location", "slug", "abbreviation")]
    wanted = normalized_name(provider_name)
    exact_aliases = {normalized_name(x) for x in names if x}
    provider_tokens = {normalized_name(x) for x in re.findall(r"[A-Za-z0-9]+", str(provider_name)) if x}
    abbreviation = normalized_name(team.get("abbreviation"))
    nickname = normalized_name(team.get("name"))
    provider_nickname = normalized_name(str(provider_name).split()[-1]) if str(provider_name).split() else ""
    return bool(
        wanted in exact_aliases or
        (abbreviation and abbreviation in provider_tokens) or
        (nickname and provider_nickname == nickname)
    )


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
        matches = []
        for team_id, team in index.items():
            # Team abbreviations and nicknames must match complete tokens. The
            # former substring check made MIA match PHI (DolPHIns), ARI match
            # CAR (CARdinals), and KC match CHI (CHIefs), discarding an entire
            # side as ambiguous even though every sportsbook supplied props.
            if espn_team_matches(team_name, team):
                matches.append((team_id, team))
        if len(matches) != 1:
            return [], []
        team_id, team = matches[0]
        league = "nfl" if sport == "nfl" else "college-football"
        def load():
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/{league}/teams/{team_id}/roster"
            return self.session.get(url, timeout=12).json()
        payload = self._cache(f"{sport}_{team_id}_roster", load, max_age=21600)
        athletes, positions = [], {}
        for group in payload.get("athletes", []):
            for athlete in group.get("items", []) if isinstance(group, dict) else []:
                name = athlete.get("fullName") or athlete.get("displayName")
                if name:
                    athletes.append(name)
                    positions[normalized_name(name)] = str(athlete.get("position", {}).get("abbreviation") or "").upper()
                    if athlete.get("id"):
                        positions[f"espn:{athlete['id']}"] = positions[normalized_name(name)]
                        positions[f"espn:{athlete['id']}__name"] = normalized_name(name)
        tokens = [team.get("abbreviation"), team.get("shortDisplayName")]
        return athletes, [x for x in tokens if x], positions

    def _mlb_roster(self, team_id):
        numeric = str(team_id or "").split(":")[-1]
        if not numeric.isdigit():
            return []
        def load():
            url = f"https://statsapi.mlb.com/api/v1/teams/{numeric}/roster"
            return self.session.get(url, params={"rosterType": "active"}, timeout=12).json()
        payload = self._cache(f"mlb_{numeric}_roster", load, max_age=21600)
        return [item.get("person", {}).get("fullName") for item in payload.get("roster", []) if item.get("person", {}).get("fullName")]

    def _mlb_schedule(self, day):
        def load():
            payload = self.session.get(
                "https://statsapi.mlb.com/api/v1/schedule",
                params={"sportId": 1, "date": day, "hydrate": "probablePitcher"}, timeout=12,
            ).json()
            return [game for date in payload.get("dates", []) for game in date.get("games", [])]
        return self._cache(f"mlb_schedule_{day}", load, max_age=900)

    def _mlb_probable_pitchers(self, event):
        start = parse_time(event.get("commence_time"))
        if not start:
            return {}
        day = start.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        away_id = str(event.get("away_team_id") or "").split(":")[-1]
        home_id = str(event.get("home_team_id") or "").split(":")[-1]
        if not away_id.isdigit() or not home_id.isdigit():
            return {}
        matches = []
        for game in self._mlb_schedule(day):
            teams = game.get("teams", {})
            if (str(teams.get("away", {}).get("team", {}).get("id")) != away_id
                    or str(teams.get("home", {}).get("team", {}).get("id")) != home_id):
                continue
            game_start = parse_time(game.get("gameDate"))
            if game_start and abs((game_start - start).total_seconds()) <= 3 * 3600:
                matches.append(game)
        if len(matches) != 1:
            return {}
        return {side: matches[0].get("teams", {}).get(side, {}).get("probablePitcher", {}).get("fullName", "")
                for side in ("away", "home")}

    def for_event(self, sport, event):
        output = {}
        if sport == "mlb":
            try:
                probables = self._mlb_probable_pitchers(event)
            except Exception:
                probables = {}
        else:
            probables = {}
        for side in ("away", "home"):
            team = str(event.get(f"{side}_team") or "")
            if sport == "mlb":
                names, tokens = self._mlb_roster(event.get(f"{side}_team_id")), []
                probable = probables.get(side)
                if probable and normalized_name(probable) not in {normalized_name(name) for name in names}:
                    names.append(probable)
            else:
                names, tokens, positions = self._espn_roster(sport, team)
            output[team], output[f"{team}__tokens"] = names, tokens
            if sport == "nfl":
                output[f"{team}__positions"] = positions
        return output

    def prefetch(self, sport, events, workers=8):
        """Bounded parallel official-roster fetch; returns event-id keyed maps."""
        events = [event for event in events if event.get("bookmakers")]
        if not events:
            return {}
        if sport in {"nfl", "ncaaf"}:
            self._espn_team_index(sport)  # one cached league index before threads
        if sport == "mlb":
            days = {start.astimezone(ZoneInfo("America/New_York")).date().isoformat()
                    for event in events if (start := parse_time(event.get("commence_time")))}
            for day in days:
                try:
                    self._mlb_schedule(day)
                except Exception:
                    pass
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


def _poll_interval_seconds(events, now, sport=None):
    leads = []
    for event in events:
        start = parse_time(event.get("commence_time"))
        if start and start > now:
            leads.append((start - now).total_seconds())
    if not leads:
        return 3600
    lead = min(leads)
    if sport == "nfl":
        if lead <= 3600:
            return 10 * 60
        if lead <= 24 * 3600:
            return 15 * 60
        return 60 * 60
    if lead <= 3600:
        return 12 * 60
    if lead <= 6 * 3600:
        return 30 * 60
    return 60 * 60


def _poll_due(last_poll, interval, now, sport):
    if not last_poll:
        return True
    # The collector runs after several sport snapshots in a five-minute cron.
    # Allow for that run-time jitter so a 15/10-minute NFL target does not
    # slip to the following cron tick (20/15 minutes in practice).
    tolerance = 90 if sport == "nfl" and interval <= 900 else 0
    return (now - last_poll).total_seconds() >= interval - tolerance


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
        interval = _poll_interval_seconds(_events(cached), now, sport)
        last_poll = parse_time(sport_state.get("last_poll"))
        due = force or _poll_due(last_poll, interval, now, sport)
        payload = cached
        if due:
            try:
                payload = client.bulk_odds(
                    config["provider_key"],
                    config["markets"] + config.get("scorer_research_markets", ()),
                )
                _atomic_json(cached_path, payload)
                state["sports"][sport] = {"last_poll": now.isoformat(), "last_success": now.isoformat()}
            except Exception as error:
                # A bad credential or provider outage must not be retried by
                # every five-minute RF runner. Record the attempt as a poll so
                # the normal event-aware interval backs errors off to hourly
                # when there is no usable cache.
                state["sports"][sport] = {
                    **sport_state,
                    "last_poll": now.isoformat(),
                    "last_error": type(error).__name__,
                    "last_error_at": now.isoformat(),
                }
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
                    "status": "UNAVAILABLE", "display_status": "Insufficient coverage",
                    "confidence": "INSUFFICIENT", "reason": f"validation_error:{type(error).__name__}",
                }
            private_lines = projection.pop("_private_lines", [])
            audit = {"projection": projection, "canonical_lines": private_lines}
            hashable_projection = {key: value for key, value in projection.items() if key not in {"generated_at", "oldest_observation_age_minutes"}}
            projection_digest = hashlib.sha256(json.dumps({"projection": hashable_projection, "canonical_lines": private_lines}, sort_keys=True, default=str).encode()).hexdigest()
            _append_changed(DATA_ROOT / "projection_ledger.jsonl", audit, projection_digest, state.get("projection_hashes", {}).get(event_key))
            state.setdefault("projection_hashes", {})[event_key] = projection_digest
            # v2 is private shadow research. It reuses the exact v1 canonical
            # lines, makes no provider request, and never enters PUBLIC_PATH.
            shadow = {}
            try:
                shadow = project_event_v2(sport, projection, private_lines, context=context, now=now)
                hashable_shadow = {key: value for key, value in shadow.items() if key != "generated_at"}
                shadow_digest = hashlib.sha256(json.dumps(hashable_shadow, sort_keys=True, default=str).encode()).hexdigest()
                _append_changed(
                    DATA_ROOT / "projection_v2_shadow_ledger.jsonl",
                    {"projection": shadow},
                    shadow_digest,
                    state.get("v2_projection_hashes", {}).get(event_key),
                )
                state.setdefault("v2_projection_hashes", {})[event_key] = shadow_digest
            except Exception as error:
                # Shadow-model defects must never affect v1 or the RF runner.
                shadow_failure = {
                    "shadow": True, "customer_facing": False, "sport": sport,
                    "event_id": str(event.get("id") or ""), "generated_at": now.isoformat(),
                    "status": "SHADOW_ERROR", "reason": type(error).__name__,
                }
                failure_hashable = {key: value for key, value in shadow_failure.items() if key != "generated_at"}
                failure_digest = hashlib.sha256(json.dumps(failure_hashable, sort_keys=True).encode()).hexdigest()
                _append_changed(
                    DATA_ROOT / "projection_v2_shadow_errors.jsonl",
                    {"projection": shadow_failure}, failure_digest,
                    state.get("v2_error_hashes", {}).get(event_key),
                )
                state.setdefault("v2_error_hashes", {})[event_key] = failure_digest
            if sport == "nfl":
                try:
                    freeze_candidate(DATA_ROOT, event, projection, private_lines, shadow, rosters, now)
                except Exception as error:
                    _append_changed(
                        DATA_ROOT / "prospective_errors.jsonl",
                        {"event_id": projection.get("event_id"), "at": now.isoformat(), "error": type(error).__name__},
                        f"{event_key}:{type(error).__name__}:{now.isoformat()}", None,
                    )
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
            state.get("v2_projection_hashes", {}).pop(event_key, None)
            state.get("v2_error_hashes", {}).pop(event_key, None)
    projections = _retain_final_pregame(state, projections, now)
    payload = {"schema_version": 1, "generated_at": now.isoformat(), "projections": projections}
    # Nginx serves only this compact projection payload. Keep credentials,
    # state, observations, and the canonical-line ledger private while making
    # the atomically replaced public file readable by the web worker.
    _atomic_json(PUBLIC_PATH, payload, mode=0o644)
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


def _shadow_digest(projection):
    hashable = {key: value for key, value in projection.items() if key != "generated_at"}
    return hashlib.sha256(json.dumps(hashable, sort_keys=True, default=str).encode()).hexdigest()


def run_shadow_backfill():
    """Build v2 history from the existing v1 canonical ledger without API calls."""
    source = _jsonl(DATA_ROOT / "projection_ledger.jsonl")
    target_path = DATA_ROOT / "projection_v2_shadow_ledger.jsonl"
    existing = _jsonl(target_path)
    existing_digests = {
        _shadow_digest(entry.get("projection", {}))
        for entry in existing
        if entry.get("projection")
    }
    state_path = DATA_ROOT / "state.json"
    state = _read_json(state_path, {})
    contexts = state.get("contexts", {})
    appended, skipped, failures = 0, 0, 0
    newest = {}
    for entry in source:
        v1 = entry.get("projection", {})
        lines = entry.get("canonical_lines", [])
        if v1.get("status") != "AVAILABLE" or not lines:
            continue
        event_key = f"{v1.get('sport')}:{v1.get('event_id')}"
        try:
            shadow = project_event_v2(v1.get("sport"), v1, lines, context=contexts.get(event_key, {}))
        except Exception:
            failures += 1
            continue
        digest = _shadow_digest(shadow)
        if digest in existing_digests:
            skipped += 1
        else:
            _append_changed(target_path, {"projection": shadow}, digest, None)
            existing_digests.add(digest)
            appended += 1
        generated = parse_time(shadow.get("generated_at"))
        prior = newest.get(event_key)
        if not prior or (generated and generated > prior[0]):
            newest[event_key] = (generated, digest)
    for event_key, (_, digest) in newest.items():
        state.setdefault("v2_projection_hashes", {})[event_key] = digest
    _atomic_json(state_path, state)
    return {"appended": appended, "skipped": skipped, "failures": failures}


def _shadow_performance(rows):
    metrics = {}
    sports = sorted({row["sport"] for row in rows})
    for sport in sports:
        sport_rows = [row for row in rows if row["sport"] == sport]
        variants = sorted({name for row in sport_rows for name in row.get("variants", {})})
        if any(row.get("v1_benchmark", {}).get("away_score") is not None for row in sport_rows):
            variants = ["v1_customer_model", *variants]
        metrics[sport] = {}
        for name in variants:
            if name == "v1_customer_model":
                samples = [
                    (row, row["v1_benchmark"])
                    for row in sport_rows
                    if row.get("v1_benchmark", {}).get("away_score") is not None
                    and row.get("v1_benchmark", {}).get("home_score") is not None
                ]
            else:
                samples = [(row, row["variants"][name]) for row in sport_rows if name in row.get("variants", {})]
            team_errors = [
                projected - actual
                for row, variant in samples
                for projected, actual in (
                    (variant["away_score"], row["actual_away"]),
                    (variant["home_score"], row["actual_home"]),
                )
            ]
            total_errors = [
                variant["away_score"] + variant["home_score"] - row["actual_away"] - row["actual_home"]
                for row, variant in samples
            ]
            margin_errors = [
                (variant["home_score"] - variant["away_score"]) - (row["actual_home"] - row["actual_away"])
                for row, variant in samples
            ]
            decisive = [
                ((variant["home_score"] > variant["away_score"]) == (row["actual_home"] > row["actual_away"]))
                for row, variant in samples
                if variant["home_score"] != variant["away_score"] and row["actual_home"] != row["actual_away"]
            ]
            metrics[sport][name] = {
                "games": len(samples),
                "team_score_mae": round(sum(abs(value) for value in team_errors) / len(team_errors), 4),
                "team_score_bias": round(sum(team_errors) / len(team_errors), 4),
                "total_mae": round(sum(abs(value) for value in total_errors) / len(total_errors), 4),
                "total_bias": round(sum(total_errors) / len(total_errors), 4),
                "margin_mae": round(sum(abs(value) for value in margin_errors) / len(margin_errors), 4),
                "decisive_winner_accuracy": round(sum(decisive) / len(decisive), 4) if decisive else None,
                "decisive_games": len(decisive),
            }
    return metrics


def run_resolution(client=None, now=None, force=False):
    """Privately join frozen pregame projections to final team scores."""
    now, client = now or utc_now(), client or PropLineClient()
    ledger = _jsonl(DATA_ROOT / "projection_ledger.jsonl")
    latest = {}
    for entry in ledger:
        projection = entry.get("projection", {})
        if projection.get("status") == "AVAILABLE":
            latest[f"{projection.get('sport')}:{projection.get('event_id')}"] = projection
    shadow_latest = {}
    for entry in _jsonl(DATA_ROOT / "projection_v2_shadow_ledger.jsonl"):
        projection = entry.get("projection", {})
        if projection.get("status") == "SHADOW_AVAILABLE":
            key = f"{projection.get('sport')}:{projection.get('event_id')}"
            prior = shadow_latest.get(key)
            if not prior or (parse_time(projection.get("generated_at")) or now) >= (parse_time(prior.get("generated_at")) or now):
                shadow_latest[key] = projection
    state_path = DATA_ROOT / "resolution_state.json"
    state = _read_json(state_path, {"sports": {}, "resolved": {}})
    shadow_state_path = DATA_ROOT / "resolution_v2_shadow_state.json"
    shadow_state = _read_json(shadow_state_path, {"resolved": {}})
    prospective_path = DATA_ROOT / "prospective_grades.json"
    prospective = _read_json(prospective_path, {"games": {}})
    prospective_pending = frozen_candidates(DATA_ROOT, now, prospective.get("games", {}))
    for sport, config in SPORTS.items():
        if not config["enabled"]:
            continue
        candidates = {key: value for key, value in latest.items() if value.get("sport") == sport and (parse_time(value.get("commence_time")) or now) < now and key not in state["resolved"]}
        shadow_candidates = {
            key: value for key, value in shadow_latest.items()
            if value.get("sport") == sport and (parse_time(value.get("commence_time")) or now) < now
            and key not in shadow_state["resolved"]
        }
        sport_prospective = {key: value for key, value in prospective_pending.items() if value.get("sport") == sport}
        if not candidates and not shadow_candidates and not sport_prospective:
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
            if key not in candidates and key not in shadow_candidates and key not in sport_prospective:
                continue
            if str(score.get("status") or "").lower() not in {"final", "completed", "complete"}:
                continue
            try:
                actual_away, actual_home = int(score["away_score"]), int(score["home_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if key in candidates:
                projection = candidates[key]
                state["resolved"][key] = {
                    "sport": sport, "event_id": projection["event_id"], "model_version": projection["model_version"],
                    "confidence": projection["confidence"], "projected_away": projection["away_score"],
                    "projected_home": projection["home_score"], "actual_away": actual_away, "actual_home": actual_home,
                    "away_error": projection["away_score"] - actual_away, "home_error": projection["home_score"] - actual_home,
                    "resolved_at": now.isoformat(),
                }
            if key in shadow_candidates:
                projection = shadow_candidates[key]
                shadow_state["resolved"][key] = {
                    "sport": sport, "event_id": projection["event_id"], "model_version": projection["model_version"],
                    "confidence": projection["confidence"], "baseline_variant": projection["baseline_variant"],
                    "projection_generated_at": projection.get("generated_at"),
                    "variants": projection["variants"], "v1_benchmark": projection.get("v1_benchmark", {}),
                    "actual_away": actual_away, "actual_home": actual_home, "resolved_at": now.isoformat(),
                }
            if key in sport_prospective:
                prospective.setdefault("games", {})[key] = grade_candidate(
                    sport_prospective[key], actual_away, actual_home, now,
                )
    rows = list(state["resolved"].values())
    metrics = {}
    for sport in SPORTS:
        errors = [error for row in rows if row["sport"] == sport for error in (row["away_error"], row["home_error"])]
        if errors:
            metrics[sport] = {"team_scores": len(errors), "mae": round(sum(abs(x) for x in errors) / len(errors), 3), "bias": round(sum(errors) / len(errors), 3)}
    _atomic_json(state_path, state)
    _atomic_json(DATA_ROOT / "performance.json", {"generated_at": now.isoformat(), "metrics": metrics, "games": rows})
    shadow_rows = list(shadow_state["resolved"].values())
    _atomic_json(shadow_state_path, shadow_state)
    _atomic_json(
        DATA_ROOT / "performance_v2_shadow.json",
        {"generated_at": now.isoformat(), "metrics": _shadow_performance(shadow_rows), "games": shadow_rows},
    )
    _atomic_json(prospective_path, prospective)
    _atomic_json(DATA_ROOT / "prospective_performance.json", performance_summary(prospective, now))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Red Fox isolated PropLine projection collector")
    parser.add_argument("command", choices=("collect", "resolve", "status", "shadow-backfill"), nargs="?", default="collect")
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
    if args.command == "shadow-backfill":
        with process_lock() as acquired:
            if not acquired:
                return 0
            print(json.dumps(run_shadow_backfill(), indent=2))
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
