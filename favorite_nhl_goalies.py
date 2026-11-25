"""Refresh confirmed NHL starters outside the board and snapshot workers."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests

from team_aliases import normalize_team_name
from nhl_release_config import OFFICIAL_FLAGS, enabled

CACHE = "favorite_nhl_goalies.json"
EVIDENCE_LOG = "favorite_nhl_goalie_observations.jsonl"
TELEMETRY_FLAG = "RED_FOX_NHL_GOALIE_TELEMETRY"
MAX_AGE = timedelta(minutes=12)
LEAGUE_CACHE = "nhl_league_schedule.json"


def matchup(away, home):
    return " @ ".join(normalize_team_name(value, sport="nhl") for value in (away, home))


def read_goalies(data_dir: Path) -> dict:
    try:
        payload = json.loads((data_dir / CACHE).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def confirmed_goalies(cache: dict, game: str, kickoff, now) -> tuple[str, str] | None:
    if " @ " not in game:
        return None
    start = pd.to_datetime(kickoff, errors="coerce", utc=True)
    if pd.isna(start):
        return None
    entries = cache.get(start.tz_convert("America/New_York").strftime("%Y%m%d"))
    if not isinstance(entries, dict):
        return None
    fetched = pd.to_datetime(entries.get("observed_at"), errors="coerce", utc=True)
    if pd.isna(fetched) or not timedelta(0) <= now - fetched <= MAX_AGE:
        return None
    games = entries.get("games")
    pair = games.get(matchup(*game.split(" @ ", 1))) if isinstance(games, dict) else None
    if isinstance(pair, list) and len(pair) == 2 and all(isinstance(name, str) and name.strip() for name in pair):
        return tuple(pair)
    return None


def goalie_evidence(cache, game, kickoff, now, *, previous_pair=None):
    """Return explicit provider evidence for paper qualification and review."""
    unavailable = {"status": "unavailable"}
    if game.count(" @ ") != 1:
        return unavailable
    start = pd.to_datetime(kickoff, errors="coerce", utc=True)
    at = pd.to_datetime(now, errors="coerce", utc=True)
    if pd.isna(start) or pd.isna(at) or at >= start:
        return unavailable
    day = cache.get(start.tz_convert("America/New_York").strftime("%Y%m%d"), {})
    observed = pd.to_datetime(day.get("observed_at"), errors="coerce", utc=True)
    if (day.get("provider_available") is not True or pd.isna(observed)
            or not timedelta(0) <= at-observed <= MAX_AGE):
        return unavailable
    evidence = day.get("evidence", {}).get(matchup(*game.split(" @ ", 1)), {})
    source_start = pd.to_datetime(evidence.get("scheduled_at"), errors="coerce", utc=True)
    if (not evidence.get("event_id") or pd.isna(source_start)
            or abs(start-source_start) > timedelta(minutes=30) or observed >= source_start):
        return unavailable
    pair = evidence.get("confirmed_pair")
    if not pair:
        return {**evidence, "status": "unconfirmed"}
    changed = bool(previous_pair and tuple(previous_pair) != tuple(pair))
    return {**evidence, "status": "verified_change" if changed else "both_confirmed"}


def _track_goalie_changes(updated, prior, data_dir):
    transitions = []
    for day, today in updated.items():
        previous = prior.get(day, {})
        if previous.get("provider_available") != today.get("provider_available"):
            transitions.append({"day": day, "status": "source_available" if today.get("provider_available")
                                else "source_unavailable", "observed_at": today.get("attempted_at", today.get("observed_at")),
                                "provider": "espn"})
        previous_evidence = previous.get("evidence", {})
        for name, record in today.get("evidence", {}).items():
            old = previous_evidence.get(name, {})
            if old.get("event_id") != record.get("event_id"):
                old = {}
            before = old.get("last_confirmed_pair")
            pair = record["confirmed_pair"]
            record["last_confirmed_pair"] = pair or before
            record["last_pair_changed_at"] = old.get("last_pair_changed_at")
            changed = bool(before and pair and before != pair)
            if changed:
                record["last_pair_changed_at"] = record["observed_at"]
            if today.get("provider_available") and (changed or old.get("starters") != record["starters"]):
                transitions.append({**record, "day": day, "matchup": name,
                                    "status": "verified_change" if changed else record["status"],
                                    "previous_confirmed_pair": before})
    if transitions:
        # The dedicated service is the sole writer. The board only reads the
        # atomic rolling cache, keeping the audit journal off the request path.
        with (paper_dir(data_dir) / EVIDENCE_LOG).open("a", encoding="utf-8") as journal:
            for row in transitions:
                journal.write(json.dumps(row, separators=(",", ":")) + "\n")


def _confirmed_day(day: str):
    endpoint = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    response = requests.get(endpoint, params={"dates": day}, timeout=(3, 7))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload.get("events"), list):
        raise ValueError("NHL scoreboard did not contain events")
    games, evidence = {}, {}
    observed = datetime.now(timezone.utc).isoformat()
    for event in payload["events"]:
        for competition in event.get("competitions", []):
            if competition.get("status", {}).get("type", {}).get("state") != "pre":
                continue
            teams, starters, details = {}, {}, {}
            for competitor in competition.get("competitors", []):
                position = competitor.get("homeAway")
                teams[position] = competitor.get("team", {}).get("displayName")
                probables = competitor.get("probables") or []
                details[position] = [
                    {"athlete_id": str((p.get("athlete") or {}).get("id") or ""),
                     "name": (p.get("athlete") or {}).get("displayName"),
                     "status": str((p.get("status") or {}).get("name") or "unknown").lower(),
                     "source_at": p.get("lastUpdated") or None}
                    for p in probables
                ]
                confirmed = [p["name"] for p in details[position] if p["status"] == "confirmed"]
                if len(confirmed) == 1 and confirmed[0]:
                    starters[position] = confirmed[0]
            if all(teams.get(key) and starters.get(key) for key in ("away", "home")):
                games[matchup(teams["away"], teams["home"])] = [starters["away"], starters["home"]]
            if all(teams.get(key) for key in ("away", "home")):
                key = matchup(teams["away"], teams["home"])
                ids = [[p["athlete_id"] for p in details.get(pos, []) if p["status"]=="confirmed"]
                       for pos in ("away", "home")]
                pair = [x[0] for x in ids] if all(len(x)==1 and x[0] for x in ids) else None
                ambiguous = key in evidence
                evidence[key] = {"provider": "espn", "event_id": None if ambiguous else str(event.get("id") or ""),
                                 "scheduled_at": competition.get("date") or event.get("date"),
                                 "observed_at": observed, "starters": details,
                                 "confirmed_pair": None if ambiguous else pair,
                                 "status": "both_confirmed" if pair and not ambiguous else "unconfirmed"}
    return {"observed_at": observed, "games": games, "evidence": evidence, "provider_available": True}


def paper_dir(data_dir):
    path = data_dir.parent / "var" / "nhl-qualification"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def _atomic(path, value):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _league_day(day):
    date = pd.Timestamp(day).date().isoformat()
    r = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}", timeout=(3, 7))
    r.raise_for_status()
    games = next(block["games"] for block in r.json()["gameWeek"] if block["date"] == date)
    return {"observed_at": datetime.now(timezone.utc).isoformat(), "games": games}


def read_league_cache(data_dir):
    try:
        return json.loads((data_dir.parent/"var"/"nhl-qualification"/LEAGUE_CACHE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def refresh(data_dir: Path) -> dict:
    now = pd.Timestamp.now(tz="America/New_York")
    dates = [(now + pd.DateOffset(days=offset)).strftime("%Y%m%d") for offset in (0, 1)]
    prior = read_goalies(data_dir)
    telemetry = enabled(TELEMETRY_FLAG)
    league_needed = (telemetry or any(enabled(flag) for flag in OFFICIAL_FLAGS.values())
                     or enabled("RED_FOX_NHL_PAPER_PUBLIC") or enabled("RED_FOX_NHL_PAPER_CROSSOVER"))
    league = {}
    updated = {}
    with ThreadPoolExecutor(max_workers=5 if league_needed else 2) as pool:
        requests_by_day = {day: pool.submit(_confirmed_day, day) for day in dates}
        league_requests = {
            day: pool.submit(_league_day, day)
            for day in [(now+pd.DateOffset(days=i)).strftime("%Y-%m-%d") for i in (-1, 0, 1)]
        } if league_needed else {}
        for day, request in requests_by_day.items():
            try:
                updated[day] = request.result()
            except (requests.RequestException, ValueError, AttributeError, TypeError):
                if day in prior:
                    updated[day] = dict(prior[day])
                if telemetry:
                    failed = updated.setdefault(day, {"games": {}, "evidence": {}})
                    failed["provider_available"] = False
                    failed["attempted_at"] = datetime.now(timezone.utc).isoformat()
                print(f"NHL goalie source unavailable for {day}; cached observations will expire.")
        if league_needed:
            prior_league = read_league_cache(data_dir)
            for day, request in league_requests.items():
                try:
                    league[day] = request.result()
                except (requests.RequestException, ValueError, KeyError, TypeError, StopIteration):
                    if day in prior_league:
                        league[day] = prior_league[day]
    data_dir.mkdir(parents=True, exist_ok=True)
    if telemetry:
        _track_goalie_changes(updated, prior, data_dir)
    else:
        updated = {day: {key: item[key] for key in ("observed_at", "games") if key in item}
                   for day, item in updated.items()}
    if league_needed:
        _atomic(paper_dir(data_dir)/LEAGUE_CACHE, league)
    _atomic(data_dir/CACHE, updated)
    return {day: len(entry.get("games", {})) for day, entry in updated.items()}


if __name__ == "__main__":
    print(refresh(Path("data")))
