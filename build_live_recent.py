"""Publish a small live-score view while preserving the final pregame board record."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import re

import pandas as pd
import requests
from team_aliases import normalize_team_name


DATA = Path("data")
OUT = DATA / "live_recent.csv"
BOARD = DATA / "anomaly_board.csv"
SNAPSHOTS = DATA / "snapshots.csv"
EMPTY_COLUMNS = ["sport", "game_id", "game", "kickoff_iso", "market_display", "flagged_side", "reaction", "path", "score_away", "score_home", "score_status", "score_state", "frozen_at_utc"]
SCOREBOARD_URLS = {
    "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "ncaaf": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
    "ncaab": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard",
    "mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
    "nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
}


def game_key(value: object, sport: str) -> str:
    value = str(value or "").replace(" vs. ", " @ ").replace(" vs ", " @ ")
    if " @ " not in value:
        return re.sub(r"[^a-z0-9]+", "", value.lower())
    away, home = value.split(" @ ", 1)
    return "@".join((normalize_team_name(away, sport), normalize_team_name(home, sport)))


def scoreboard(sport: str, now: datetime) -> dict[str, list[dict[str, str]]]:
    base = SCOREBOARD_URLS.get(sport)
    if not base:
        return {}
    extra = "&groups=80&limit=500" if sport == "ncaaf" else "&groups=50&limit=500" if sport == "ncaab" else "&limit=500"
    games: dict[str, list[dict[str, str]]] = {}
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
                "event_time": str(event.get("date", "")),
            }
            games.setdefault(game_key(f"{away_name} @ {home_name}", sport), []).append(item)
    return games


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() and path.stat().st_size else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def market_type(side: object) -> str:
    value = str(side or "").strip().lower()
    if value.startswith(("over ", "under ")):
        return "TOTAL"
    return "SPREAD" if re.search(r"\s[+-]\d+(?:\.\d+)?(?:\s|$)", value) else "MONEYLINE"


def bootstrap_started_records(now: datetime) -> pd.DataFrame:
    """Recover the active window if an earlier pregame handoff was interrupted."""
    snapshots = read_csv_or_empty(SNAPSHOTS)
    if snapshots.empty:
        return pd.DataFrame()
    snapshots["_kickoff"] = pd.to_datetime(snapshots.get("dk_start_iso", ""), errors="coerce", utc=True)
    snapshots["_seen"] = pd.to_datetime(snapshots.get("timestamp", ""), errors="coerce", utc=True)
    window = snapshots[snapshots["_kickoff"].notna() & (snapshots["_kickoff"] <= now) & (snapshots["_kickoff"] >= now - timedelta(hours=10))].copy()
    window = window[window["_seen"] <= window["_kickoff"]].copy()
    if window.empty:
        return pd.DataFrame()
    window["market_display"] = window["side"].map(market_type)
    latest = window.sort_values("_seen").groupby(["sport", "game_id", "market_display", "side"], as_index=False).tail(1).copy()
    latest["_split_gap"] = (pd.to_numeric(latest["money_pct"], errors="coerce") - pd.to_numeric(latest["bets_pct"], errors="coerce")).abs()
    latest = latest.sort_values("_split_gap", ascending=False).drop_duplicates(["sport", "game_id", "market_display"], keep="first")
    latest = latest.rename(columns={"side": "flagged_side", "dk_start_iso": "kickoff_iso"})
    latest["reaction"] = "Observed"
    latest["path"] = "Pregame snapshot"
    latest["reason"] = "Frozen from the final available pregame snapshot."
    latest["frozen_at_utc"] = now.isoformat()
    return latest


def main(scores_only: bool = False) -> None:
    now = datetime.now(timezone.utc)
    existing = read_csv_or_empty(OUT)
    # The one-minute worker also reads the already-published pregame board so
    # a record is frozen at kickoff instead of waiting for the next snapshot run.
    previous = read_csv_or_empty(BOARD)
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
        recovered = bootstrap_started_records(now)
        if recovered.empty:
            pd.DataFrame(columns=EMPTY_COLUMNS).to_csv(OUT, index=False)
            return
        rows.append(recovered)
    live = pd.concat(rows, ignore_index=True, sort=False)
    # Compare in one timezone-free representation; CSVs can contain a mix of
    # offset-aware and legacy naive kickoff values.
    kickoff_values = live["kickoff_iso"] if "kickoff_iso" in live.columns else pd.Series(pd.NaT, index=live.index)
    kickoff_parsed = pd.to_datetime(kickoff_values, errors="coerce", utc=True)
    live["_kickoff"] = kickoff_parsed.dt.tz_localize(None) if isinstance(kickoff_parsed, pd.Series) else pd.Series(pd.NaT, index=live.index)
    # Keep this as a short look-in, not a results archive. Final games stay
    # available for the rest of the day; an unresolved live status gets a small safety window.
    state = live.get("score_state", pd.Series("", index=live.index)).astype(str).str.lower()
    now_naive = pd.Timestamp(now).tz_localize(None)
    cutoff_final = now_naive - pd.Timedelta(hours=10)
    cutoff_unresolved = now_naive - pd.Timedelta(hours=8)
    final = state.eq("post")
    live = live[live["_kickoff"].notna() & (((final) & (live["_kickoff"] >= cutoff_final)) | ((~final) & (live["_kickoff"] >= cutoff_unresolved)))].copy()
    key_columns = [column for column in ("sport", "game_id", "market_display", "flagged_side") if column in live]
    if key_columns:
        live = live.sort_values("frozen_at_utc", na_position="last").drop_duplicates(key_columns, keep="first")
    if scores_only:
        for sport, indices in live.groupby("sport").groups.items():
            scores = scoreboard(str(sport).lower(), now)
            for index in indices:
                candidates = scores.get(game_key(live.at[index, "game"], str(sport).lower()), [])
                kickoff = pd.to_datetime(live.at[index, "kickoff_iso"], errors="coerce", utc=True)
                score = min(candidates, key=lambda item: abs(pd.to_datetime(item["event_time"], errors="coerce", utc=True) - kickoff)) if candidates and pd.notna(kickoff) else None
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
