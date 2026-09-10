"""Publish a small live-score view while preserving the final pregame board record."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import json
import re

import pandas as pd
import requests
from team_aliases import normalize_team_name


DATA = Path("data")
OUT = DATA / "live_recent.csv"
SCORE_COVERAGE_OUT = DATA / "live_score_coverage.json"
BOARD = DATA / "anomaly_board.csv"
SNAPSHOTS = DATA / "snapshots.csv"
FINAL_RETENTION_HOURS = 10
UNRESOLVED_RETENTION_HOURS = 8
SCORE_STALE_MINUTES = 3
EMPTY_COLUMNS = [
    "sport", "game_id", "game", "kickoff_iso", "market_display", "flagged_side", "reaction", "path",
    "score_away", "score_home", "score_status", "score_state", "score_provider", "score_provider_event_id",
    "score_match_state", "score_updated_at_utc", "score_completed_at_utc", "frozen_at_utc",
    "red_fox_favorite", "favorite_side", "favorite_pathway", "favorite_rule_version",
    "favorite_first_qualified_at", "favorite_final_qualified_at", "favorite_state",
    "favorite_final_market_read", "favorite_final_market_rank",
    "favorite_supporting_evidence", "favorite_whipsaw_state", "favorite_cross_market_state",
    "favorite_snapshot_id", "favorite_reason",
]
SCOREBOARD_URLS = {
    "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "ncaaf": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
    "ncaab": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard",
    "mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
    "nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
}

# ESPN occasionally spells an MLB city out where the sportsbook uses its three
# letter abbreviation. Keep these score-feed-only aliases explicit.
SCOREBOARD_TEAM_ALIASES = {
    "chi white sox": "chicago white sox",
    "chi cubs": "chicago cubs",
    "la angels": "los angeles angels",
    "ny mets": "new york mets",
    "ny yankees": "new york yankees",
    # Score-feed-only college identities observed in the live inventory.
    "mississippi valley": "mississippi valley st",
    "mvsu": "mississippi valley st",
    "ulm": "ul monroe",
}
COMPOUND_NICKNAMES = (
    "blue jackets", "blue jays", "golden knights", "maple leafs",
    "red sox", "red wings", "trail blazers", "white sox",
)


def game_key(value: object, sport: str) -> str:
    value = str(value or "").replace(" vs. ", " @ ").replace(" vs ", " @ ")
    if " @ " not in value:
        return re.sub(r"[^a-z0-9]+", "", value.lower())
    away, home = value.split(" @ ", 1)
    away_key = normalize_team_name(away, sport)
    home_key = normalize_team_name(home, sport)
    away_key = SCOREBOARD_TEAM_ALIASES.get(away_key, away_key)
    home_key = SCOREBOARD_TEAM_ALIASES.get(home_key, home_key)
    return "@".join((away_key, home_key))


def team_signature(value: object, sport: str) -> str:
    """Return a narrow nickname signature for a paired-match fallback only."""
    normalized = normalize_team_name(str(value or ""), sport)
    normalized = SCOREBOARD_TEAM_ALIASES.get(normalized, normalized)
    for nickname in COMPOUND_NICKNAMES:
        if normalized.endswith(nickname):
            return nickname
    return normalized.rsplit(" ", 1)[-1]


def signature_game_key(value: object, sport: str) -> str:
    value = str(value or "").replace(" vs. ", " @ ").replace(" vs ", " @ ")
    if " @ " not in value:
        return ""
    away, home = value.split(" @ ", 1)
    return "@".join((team_signature(away, sport), team_signature(home, sport)))


def provider_team_names(team: dict, sport: str) -> set[str]:
    """Return only provider-authored team identities for exact paired matching."""
    names = {
        str(team.get(field, "")).strip()
        for field in ("displayName", "location", "shortDisplayName", "abbreviation")
    }
    normalized = {normalize_team_name(name, sport) for name in names if name}
    return {SCOREBOARD_TEAM_ALIASES.get(name, name) for name in normalized}


def provider_game_keys(away_team: dict, home_team: dict, sport: str) -> set[str]:
    return {
        f"{away}@{home}"
        for away in provider_team_names(away_team, sport)
        for home in provider_team_names(home_team, sport)
        if away and home
    }


def fetch_scoreboard(sport: str, now: datetime) -> tuple[dict[str, list[dict[str, str]]], str]:
    base = SCOREBOARD_URLS.get(sport)
    if not base:
        return {}, "unsupported"
    extra = "&groups=80&limit=500" if sport == "ncaaf" else "&groups=50&limit=500" if sport == "ncaab" else "&limit=500"
    games: dict[str, list[dict[str, str]]] = {}
    fetched = False
    seen_events: set[str] = set()
    for day in (now - timedelta(days=1), now):
        try:
            response = requests.get(f"{base}?dates={day:%Y%m%d}{extra}", timeout=12)
            response.raise_for_status()
            events = response.json().get("events", [])
            fetched = True
        except Exception as error:
            print(f"[live-recent] {sport} scoreboard unavailable: {type(error).__name__}")
            continue
        for event in events:
            event_id = str(event.get("id", ""))
            if event_id and event_id in seen_events:
                continue
            if event_id:
                seen_events.add(event_id)
            competition = (event.get("competitions") or [{}])[0]
            competitors = competition.get("competitors") or []
            away = next((item for item in competitors if item.get("homeAway") == "away"), {})
            home = next((item for item in competitors if item.get("homeAway") == "home"), {})
            away_team = away.get("team") or {}
            home_team = home.get("team") or {}
            away_name = away_team.get("displayName", "")
            home_name = home_team.get("displayName", "")
            status = event.get("status") or {}
            status_type = status.get("type") or {}
            detail = status_type.get("shortDetail") or status_type.get("detail") or "In progress"
            item = {
                "score_away": str(away.get("score", "-")),
                "score_home": str(home.get("score", "-")),
                "score_status": detail,
                "score_state": str(status_type.get("state", "in")),
                "event_time": str(event.get("date", "")),
                "score_provider": "espn",
                "score_provider_event_id": event_id,
            }
            matchup = f"{away_name} @ {home_name}"
            # Exact canonical names are primary. Provider location/short-name/
            # abbreviation pairs safely cover FBS/FCS mascot and acronym drift.
            keys = {game_key(matchup, sport), signature_game_key(matchup, sport)}
            keys.update(provider_game_keys(away_team, home_team, sport))
            for key in keys:
                if key:
                    games.setdefault(key, []).append(item)
    return games, "available" if fetched else "unavailable"


def scoreboard(sport: str, now: datetime) -> dict[str, list[dict[str, str]]]:
    """Backward-compatible score map used by existing diagnostics."""
    return fetch_scoreboard(sport, now)[0]


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() and path.stat().st_size else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def ensure_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the frozen export schema stable across pre-feature retained rows."""
    result = frame.copy()
    for column in EMPTY_COLUMNS:
        if column not in result.columns:
            result[column] = "false" if column == "red_fox_favorite" else ""
    return result


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


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def write_score_coverage(live: pd.DataFrame, now: datetime) -> dict:
    games = live.drop_duplicates([column for column in ("sport", "game_id", "game") if column in live]).copy()
    state = games.get("score_state", pd.Series("", index=games.index)).astype(str).str.lower()
    active = games[~state.eq("post")].copy()
    match = active.get("score_match_state", pd.Series("unmatched", index=active.index)).astype(str).str.lower()
    updated = pd.to_datetime(active.get("score_updated_at_utc", ""), errors="coerce", utc=True)
    stale = match.eq("matched") & (updated.isna() | (updated < pd.Timestamp(now) - pd.Timedelta(minutes=SCORE_STALE_MINUTES)))
    has_score = (
        active.get("score_away", pd.Series("", index=active.index)).astype(str).ne("-")
        & active.get("score_home", pd.Series("", index=active.index)).astype(str).ne("-")
    )
    payload = {
        "generated_at_utc": now.isoformat(),
        "retention": {"final_hours": FINAL_RETENTION_HOURS, "unresolved_hours": UNRESOLVED_RETENTION_HOURS},
        "active_live_games": int(len(active)),
        "matched": int(match.eq("matched").sum()),
        "receiving_score": int((match.eq("matched") & has_score & ~stale).sum()),
        "unmatched": int(match.eq("unmatched").sum()),
        "stale": int(stale.sum()),
        "provider_unavailable": int(match.isin(["provider_unavailable", "unsupported"]).sum()),
        "games": [
            {
                "sport": _text(row.get("sport")), "game_id": _text(row.get("game_id")),
                "game": _text(row.get("game")), "state": _text(row.get("score_state")),
                "coverage": _text(row.get("score_match_state")),
                "provider_event_id": _text(row.get("score_provider_event_id")),
                "updated_at_utc": _text(row.get("score_updated_at_utc")),
            }
            for _, row in active.iterrows()
        ],
    }
    SCORE_COVERAGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = SCORE_COVERAGE_OUT.with_name("." + SCORE_COVERAGE_OUT.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(SCORE_COVERAGE_OUT)
    print(
        "[live-recent] score coverage: "
        f"{payload['matched']}/{payload['active_live_games']} matched, "
        f"{payload['receiving_score']}/{payload['active_live_games']} current, "
        f"{payload['unmatched']} unmatched, {payload['stale']} stale, "
        f"{payload['provider_unavailable']} provider unavailable"
    )
    return payload


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
            empty = pd.DataFrame(columns=EMPTY_COLUMNS)
            empty.to_csv(OUT, index=False)
            write_score_coverage(empty, now)
            return
        rows.append(recovered)
    live = ensure_output_columns(pd.concat(rows, ignore_index=True, sort=False))
    # Compare in one timezone-free representation; CSVs can contain a mix of
    # offset-aware and legacy naive kickoff values.
    kickoff_values = live["kickoff_iso"] if "kickoff_iso" in live.columns else pd.Series(pd.NaT, index=live.index)
    kickoff_parsed = pd.to_datetime(kickoff_values, errors="coerce", utc=True)
    live["_kickoff"] = kickoff_parsed.dt.tz_localize(None) if isinstance(kickoff_parsed, pd.Series) else pd.Series(pd.NaT, index=live.index)
    # Keep this as a short look-in, not a results archive. Final games stay
    # available for the rest of the day; an unresolved live status gets a small safety window.
    state = live.get("score_state", pd.Series("", index=live.index)).astype(str).str.lower()
    now_naive = pd.Timestamp(now).tz_localize(None)
    cutoff_final = now_naive - pd.Timedelta(hours=FINAL_RETENTION_HOURS)
    cutoff_unresolved = now_naive - pd.Timedelta(hours=UNRESOLVED_RETENTION_HOURS)
    final = state.eq("post")
    live = live[live["_kickoff"].notna() & (((final) & (live["_kickoff"] >= cutoff_final)) | ((~final) & (live["_kickoff"] >= cutoff_unresolved)))].copy()
    key_columns = [column for column in ("sport", "game_id", "market_display", "flagged_side") if column in live]
    if key_columns:
        live = live.sort_values("frozen_at_utc", na_position="last").drop_duplicates(key_columns, keep="first")
    if scores_only:
        for sport, indices in live.groupby("sport").groups.items():
            scores, provider_state = fetch_scoreboard(str(sport).lower(), now)
            for index in indices:
                sport_key = str(sport).lower()
                matchup = live.at[index, "game"]
                candidates = scores.get(game_key(matchup, sport_key), [])
                if not candidates:
                    candidates = scores.get(signature_game_key(matchup, sport_key), [])
                kickoff = pd.to_datetime(live.at[index, "kickoff_iso"], errors="coerce", utc=True)
                score = min(candidates, key=lambda item: abs(pd.to_datetime(item["event_time"], errors="coerce", utc=True) - kickoff)) if candidates and pd.notna(kickoff) else None
                if score:
                    previous_state = _text(live.at[index, "score_state"] if "score_state" in live else "").lower()
                    for column, value in score.items():
                        live.at[index, column] = value
                    live.at[index, "score_match_state"] = "matched"
                    live.at[index, "score_updated_at_utc"] = now.isoformat()
                    if str(score.get("score_state", "")).lower() == "post" and previous_state != "post":
                        live.at[index, "score_completed_at_utc"] = now.isoformat()
                else:
                    if provider_state == "available":
                        live.at[index, "score_away"] = "-"
                        live.at[index, "score_home"] = "-"
                        live.at[index, "score_status"] = "Live score unavailable"
                        live.at[index, "score_state"] = "unknown"
                        live.at[index, "score_match_state"] = "unmatched"
                    else:
                        # Preserve the last known score through a transient source
                        # failure. Coverage marks it unavailable/stale explicitly.
                        live.at[index, "score_match_state"] = "unsupported" if provider_state == "unsupported" else "provider_unavailable"
                        if not _text(live.at[index, "score_status"] if "score_status" in live else ""):
                            live.at[index, "score_status"] = "Score unavailable"
    live = live.drop(columns=["_kickoff"], errors="ignore").sort_values("kickoff_iso", ascending=False)
    temp = DATA / ".live_recent.csv.tmp"
    live.to_csv(temp, index=False)
    temp.replace(OUT)
    write_score_coverage(live, now)
    print(f"[live-recent] wrote {len(live)} frozen pregame records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-only", action="store_true", help="Refresh scores without touching frozen board records")
    main(parser.parse_args().scores_only)
