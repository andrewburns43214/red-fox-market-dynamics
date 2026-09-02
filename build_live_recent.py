"""Publish a small live-score view while preserving the final pregame board record."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import re

import pandas as pd
import requests


DATA = Path("data")
OUT = DATA / "live_recent.csv"
BOARD = DATA / "anomaly_board.csv"
SCOREBOARD_URLS = {
    "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "ncaaf": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
    "ncaab": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard",
    "mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
    "nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
}


def norm(value: object) -> str:
    value = str(value or "").lower().replace(" vs. ", " @ ").replace(" vs ", " @ ")
    return re.sub(r"[^a-z0-9]+", "", value)


def scoreboard(sport: str, now: datetime) -> dict[str, dict[str, str]]:
    base = SCOREBOARD_URLS.get(sport)
    if not base:
        return {}
    extra = "&groups=80&limit=500" if sport == "ncaaf" else "&groups=50&limit=500" if sport == "ncaab" else "&limit=500"
    games: dict[str, dict[str, str]] = {}
    for day in (now - timedelta(days=1), now):
        try:
            response = requests.get(f"{base}?dates={day:%Y%m%d}{extra}", timeout=12)
            response.raise_for_status()
            events = response.json().get("events", [])
        except Exception as error:
            print(f"[live-recent] {sport} scoreboard unavailable: {type(error).__name__}")
            continue
        for event in events:
            competition = (event.get("competitions") or [{}])[0]
            competitors = competition.get("competitors") or []
            away = next((item for item in competitors if item.get("homeAway") == "away"), {})
            home = next((item for item in competitors if item.get("homeAway") == "home"), {})
            away_name = (away.get("team") or {}).get("displayName", "")
            home_name = (home.get("team") or {}).get("displayName", "")
            status = event.get("status") or {}
            status_type = status.get("type") or {}
            detail = status_type.get("shortDetail") or status_type.get("detail") or "In progress"
            item = {
                "score_away": str(away.get("score", "-")),
                "score_home": str(home.get("score", "-")),
                "score_status": detail,
                "score_state": str(status_type.get("state", "in")),
            }
            games[norm(f"{away_name} @ {home_name}")] = item
    return games


def main(scores_only: bool = False) -> None:
    now = datetime.now(timezone.utc)
    existing = pd.read_csv(OUT, dtype=str, keep_default_na=False) if OUT.exists() else pd.DataFrame()
    previous = pd.DataFrame() if scores_only else (pd.read_csv(BOARD, dtype=str, keep_default_na=False) if BOARD.exists() else pd.DataFrame())
    rows = []
    if not existing.empty:
        rows.append(existing)
    if not previous.empty and "kickoff_iso" in previous:
        prior = previous.copy()
        prior["_kickoff"] = pd.to_datetime(prior["kickoff_iso"], errors="coerce", utc=True)
        started = prior[prior["_kickoff"].notna() & (prior["_kickoff"] <= now)].copy()
        if not started.empty:
            started["frozen_at_utc"] = now.isoformat()
            rows.append(started.drop(columns=["_kickoff"]))
    if not rows:
        pd.DataFrame().to_csv(OUT, index=False)
        return
    live = pd.concat(rows, ignore_index=True, sort=False)
    live["_kickoff"] = pd.to_datetime(live.get("kickoff_iso", ""), errors="coerce", utc=True)
    # Keep only the current look-in window. This is not a results archive.
    live = live[live["_kickoff"].notna() & (live["_kickoff"] >= now - timedelta(hours=18))].copy()
    key_columns = [column for column in ("sport", "game_id", "market_display", "flagged_side") if column in live]
    if key_columns:
        live = live.sort_values("frozen_at_utc", na_position="last").drop_duplicates(key_columns, keep="first")
    if scores_only:
        for sport, indices in live.groupby("sport").groups.items():
            scores = scoreboard(str(sport).lower(), now)
            for index in indices:
                score = scores.get(norm(live.at[index, "game"]))
                if score:
                    for column, value in score.items():
                        live.at[index, column] = value
                else:
                    live.at[index, "score_away"] = "-"
                    live.at[index, "score_home"] = "-"
                    live.at[index, "score_status"] = "Live score unavailable"
                    live.at[index, "score_state"] = "unknown"
    live = live.drop(columns=["_kickoff"], errors="ignore").sort_values("kickoff_iso", ascending=False)
    temp = DATA / ".live_recent.csv.tmp"
    live.to_csv(temp, index=False)
    temp.replace(OUT)
    print(f"[live-recent] wrote {len(live)} frozen pregame records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-only", action="store_true", help="Refresh scores without touching frozen board records")
    main(parser.parse_args().scores_only)
