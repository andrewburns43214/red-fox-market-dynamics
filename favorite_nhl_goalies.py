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

CACHE = "favorite_nhl_goalies.json"
MAX_AGE = timedelta(minutes=12)


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


def _confirmed_day(day: str):
    endpoint = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    response = requests.get(endpoint, params={"dates": day}, timeout=(3, 7))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload.get("events"), list):
        raise ValueError("NHL scoreboard did not contain events")
    games = {}
    for event in payload["events"]:
        for competition in event.get("competitions", []):
            if competition.get("status", {}).get("type", {}).get("state") != "pre":
                continue
            teams, starters = {}, {}
            for competitor in competition.get("competitors", []):
                position = competitor.get("homeAway")
                teams[position] = competitor.get("team", {}).get("displayName")
                confirmed = [
                    probable.get("athlete", {}).get("displayName")
                    for probable in competitor.get("probables", [])
                    if str((probable.get("status") or {}).get("name", "")).lower() == "confirmed"
                ]
                if len(confirmed) == 1 and confirmed[0]:
                    starters[position] = confirmed[0]
            if all(teams.get(key) and starters.get(key) for key in ("away", "home")):
                games[matchup(teams["away"], teams["home"])] = [starters["away"], starters["home"]]
    return {"observed_at": datetime.now(timezone.utc).isoformat(), "games": games}


def refresh(data_dir: Path) -> dict:
    now = pd.Timestamp.now(tz="America/New_York")
    dates = [(now + pd.DateOffset(days=offset)).strftime("%Y%m%d") for offset in (0, 1)]
    prior = read_goalies(data_dir)
    updated = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        requests_by_day = {day: pool.submit(_confirmed_day, day) for day in dates}
        for day, request in requests_by_day.items():
            try:
                updated[day] = request.result()
            except (requests.RequestException, ValueError, AttributeError, TypeError):
                if day in prior:
                    updated[day] = prior[day]
                print(f"NHL goalie source unavailable for {day}; cached observations will expire.")
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / CACHE
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(updated, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {day: len(entry.get("games", {})) for day, entry in updated.items()}


if __name__ == "__main__":
    print(refresh(Path("data")))
