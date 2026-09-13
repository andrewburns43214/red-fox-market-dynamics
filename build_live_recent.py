"""Publish a small live-score view while preserving the final pregame board record."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import json
import re

import pandas as pd
import requests
from red_fox_favorite import FAVORITE_COLUMNS, VISIBILITY_INVALIDATED_FAVORITES
from team_aliases import normalize_team_name


DATA = Path("data")
OUT = DATA / "live_recent.csv"
SCORE_COVERAGE_OUT = DATA / "live_score_coverage.json"
BOARD = DATA / "anomaly_board.csv"
SNAPSHOTS = DATA / "snapshots.csv"
FAVORITE_CANDIDATES = DATA / "red_fox_favorite_freeze_candidates.csv"
MARKET_CANDIDATES = DATA / "live_recent_market_candidates.csv"
FINAL_RETENTION_HOURS = 10
UNRESOLVED_RETENTION_HOURS = 8
SCORE_STALE_MINUTES = 3
EMPTY_COLUMNS = [
    "sport", "game_id", "game", "kickoff_iso", "market_display", "flagged_side", "reaction", "path",
    "score_away", "score_home", "score_status", "score_state", "score_provider", "score_provider_event_id",
    "score_match_state", "score_updated_at_utc", "score_completed_at_utc", "frozen_at_utc",
    "state_as_of_utc", "final_pregame_state_at_utc", "freeze_method",
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
    # DraftKings uses the university name while ESPN brands the athletics
    # program as "App State".  Normalize the provider spelling to the same
    # canonical identity used by team_aliases.py.
    "app state": "appalachian st",
    "chi white sox": "chicago white sox",
    "chi cubs": "chicago cubs",
    "la angels": "los angeles angels",
    "ny mets": "new york mets",
    "ny yankees": "new york yankees",
    # Score-feed-only college identities observed in the live inventory.
    "mississippi valley": "mississippi valley st",
    "mvsu": "mississippi valley st",
    "southern university": "southern",
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
    market_keys = ["sport", "game_id", "market_display"]
    latest_seen = window.groupby(market_keys)["_seen"].transform("max")
    paired = window.loc[window["_seen"].eq(latest_seen)].copy()
    paired["_split_gap"] = (pd.to_numeric(paired["money_pct"], errors="coerce") - pd.to_numeric(paired["bets_pct"], errors="coerce")).abs()
    latest = paired.sort_values("_split_gap", ascending=False).drop_duplicates(market_keys, keep="first").copy()
    pair_map = {}
    for key, sides in paired.groupby(market_keys, sort=False):
        pair_map[tuple(str(value) for value in key)] = json.dumps([
            {
                "flagged_side": side.get("side", ""), "bets_pct": side.get("bets_pct", ""),
                "money_pct": side.get("money_pct", ""), "open_line": side.get("open_line", ""),
                "current_line": side.get("current_line", ""), "reaction": "Observed",
                "path": "Pregame snapshot", "anomaly_chips": "Observed pregame market",
            }
            for _, side in sides.iterrows()
        ], separators=(",", ":"))
    latest["market_sides"] = latest.apply(
        lambda row: pair_map.get(tuple(str(row.get(column, "")) for column in market_keys), "[]"), axis=1
    )
    latest = latest.rename(columns={"side": "flagged_side", "dk_start_iso": "kickoff_iso"})
    latest["reaction"] = "Observed"
    latest["path"] = "Pregame snapshot"
    latest["reason"] = "Frozen from the final available pregame snapshot."
    latest["frozen_at_utc"] = now.isoformat()
    latest["state_as_of_utc"] = latest["_seen"].map(lambda value: value.isoformat() if pd.notna(value) else "")
    latest["final_pregame_state_at_utc"] = latest["state_as_of_utc"]
    # This recovery path has raw market data only.  It is useful for the live
    # score screen but never reconstructs Red Fox classifications.
    latest["freeze_method"] = "raw_snapshot_recovery_no_classification"
    return latest


def update_market_candidates(rows: pd.DataFrame, path: Path = MARKET_CANDIDATES, as_of=None) -> pd.DataFrame:
    """Retain a compact, paired all-market handoff for Live & Recent."""
    existing = read_csv_or_empty(path)
    if rows is None or rows.empty:
        return existing
    work = rows.copy()
    required = {"sport", "game_id", "game", "market_display", "side", "bets_pct", "money_pct", "open_line", "current_line", "dk_start_iso"}
    if not required.issubset(work.columns):
        return existing
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    work["_kickoff"] = pd.to_datetime(work["dk_start_iso"], errors="coerce", utc=True)
    seen_values = work["timestamp"] if "timestamp" in work else pd.Series(now, index=work.index)
    work["_seen"] = pd.to_datetime(seen_values, errors="coerce", utc=True)
    work = work[work["_kickoff"].notna() & work["_seen"].notna() & (work["_seen"] <= work["_kickoff"])].copy()
    if work.empty:
        return existing
    keys = ["sport", "game_id", "market_display"]
    # Callers normally provide one synchronized capture, but selecting the
    # latest timestamp here makes the handoff safe for a raw-history backfill.
    latest_seen = work.groupby(keys)["_seen"].transform("max")
    work = work.loc[work["_seen"].eq(latest_seen)].copy()
    records = []
    for _, sides in work.groupby(keys, sort=False):
        if sides["side"].astype(str).nunique() != 2:
            continue
        representative = sides.assign(
            _gap=(pd.to_numeric(sides["money_pct"], errors="coerce") - pd.to_numeric(sides["bets_pct"], errors="coerce")).abs()
        ).sort_values("_gap", ascending=False).iloc[0].copy()
        representative["flagged_side"] = representative.get("side", "")
        representative["kickoff_iso"] = representative.get("dk_start_iso", "")
        representative["reaction"] = "Observed"
        representative["path"] = "Pregame snapshot"
        representative["reason"] = "Frozen from the final available paired pregame market."
        representative["state_as_of_utc"] = representative["_seen"].isoformat()
        representative["final_pregame_state_at_utc"] = representative["state_as_of_utc"]
        representative["freeze_method"] = "paired_raw_market_handoff_no_classification"
        representative["red_fox_favorite"] = "false"
        representative["favorite_state"] = "not_qualified"
        representative["market_sides"] = json.dumps([
            {
                "flagged_side": side.get("side", ""), "bets_pct": side.get("bets_pct", ""),
                "money_pct": side.get("money_pct", ""), "open_line": side.get("open_line", ""),
                "current_line": side.get("current_line", ""), "reaction": "Observed",
                "path": "Pregame snapshot", "anomaly_chips": "Observed pregame market",
            }
            for _, side in sides.iterrows()
        ], separators=(",", ":"))
        records.append(representative.drop(labels=["_gap", "_kickoff", "_seen"], errors="ignore"))
    if not records:
        return existing
    current = pd.DataFrame(records)
    updated = current if existing.empty else pd.concat([existing, current], ignore_index=True, sort=False)
    updated["_state"] = pd.to_datetime(updated.get("state_as_of_utc", ""), errors="coerce", utc=True)
    updated["_kickoff"] = pd.to_datetime(updated.get("kickoff_iso", ""), errors="coerce", utc=True)
    # The handoff only bridges the pregame board into the short Live & Recent
    # window. Prune old candidates so this compact safety cache stays bounded.
    updated = updated[updated["_kickoff"].isna() | (updated["_kickoff"] >= now - pd.Timedelta(hours=12))].copy()
    updated = updated.sort_values("_state", kind="mergesort").drop_duplicates(keys, keep="last")
    updated = updated.drop(columns=["_state", "_kickoff"], errors="ignore")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    updated.to_csv(temporary, index=False)
    temporary.replace(path)
    return updated


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def utc_series(values: object, index: pd.Index) -> pd.Series:
    """Return an explicitly UTC-aware Series, including all-NaT inputs."""
    source = values if isinstance(values, pd.Series) else pd.Series(values, index=index)
    parsed = pd.Series(pd.to_datetime(source, errors="coerce", utc=True), index=index)
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize("UTC")
    return parsed


def final_pregame_states(previous: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Return started rows whose published source state is valid at kickoff."""
    if previous.empty or "kickoff_iso" not in previous:
        return pd.DataFrame()
    prior = previous.copy()
    prior["_kickoff"] = utc_series(prior["kickoff_iso"], prior.index)
    prior["_state_at"] = utc_series(prior.get("state_as_of_utc", ""), prior.index)
    started = prior[
        prior["_kickoff"].notna()
        & (prior["_kickoff"] <= now)
        & prior["_state_at"].notna()
        & (prior["_state_at"] <= prior["_kickoff"])
    ].copy()
    if started.empty:
        return started.drop(columns=["_kickoff", "_state_at"], errors="ignore")
    started["frozen_at_utc"] = now.isoformat()
    started["final_pregame_state_at_utc"] = started["_state_at"].map(lambda value: value.isoformat())
    started["freeze_method"] = "published_state_latest_at_or_before_start"
    return started.drop(columns=["_kickoff", "_state_at"])


def favorite_handoff_states(candidates: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Freeze previously qualified Favorites even if the live board drops them.

    Qualification is an auditable event, while board presence depends on a
    short source-freshness window.  Keeping those concepts separate prevents a
    single missed scrape from erasing the Favorite at kickoff.
    """
    if candidates.empty:
        return pd.DataFrame()
    candidates = apply_favorite_exclusions(candidates)
    favorite = candidates[
        candidates.get("red_fox_favorite", pd.Series("false", index=candidates.index))
        .astype(str).str.lower().isin({"1", "true", "yes"})
    ].copy()
    frozen = final_pregame_states(favorite, now)
    if not frozen.empty:
        frozen["freeze_method"] = "favorite_tracking_last_qualified_at_or_before_start"
    return frozen


def apply_favorite_exclusions(frame: pd.DataFrame) -> pd.DataFrame:
    """Prevent audited visibility failures from re-entering frozen displays."""
    if frame.empty or not all(column in frame for column in ("sport", "game_id", "market_display")):
        return frame
    result = frame.copy()
    invalid = pd.Series(
        [
            (str(row.get("sport", "")).lower(), str(row.get("game_id", "")), str(row.get("market_display", "")).upper())
            in VISIBILITY_INVALIDATED_FAVORITES
            for _, row in result.iterrows()
        ],
        index=result.index,
    )
    for column in FAVORITE_COLUMNS:
        if column in result:
            result.loc[invalid, column] = ""
    if "red_fox_favorite" in result:
        result.loc[invalid, "red_fox_favorite"] = "false"
    if "favorite_state" in result:
        result.loc[invalid, "favorite_state"] = "not_qualified"
    if "favorite_reason" in result:
        result.loc[invalid, "favorite_reason"] = "Favorite removed: audited final-hour visibility failure."
    return result


def expire_started_board_rows(now: datetime, grace_minutes: int = 5) -> int:
    """Atomically remove started games from the pregame board.

    The full board publisher can take several minutes on a large Saturday
    slate.  Its kickoff gate must not depend on that expensive rebuild reaching
    the final write.  The one-minute Live & Recent worker calls this only after
    it has had an opportunity to freeze the existing board rows.
    """
    board = read_csv_or_empty(BOARD)
    if board.empty or "kickoff_iso" not in board.columns:
        return 0
    kickoff = utc_series(board["kickoff_iso"], board.index)
    cutoff = pd.Timestamp(now) - pd.Timedelta(minutes=grace_minutes)
    keep = kickoff.isna() | (kickoff > cutoff)
    removed = int((~keep).sum())
    if not removed:
        return 0
    temporary = BOARD.with_name("." + BOARD.name + ".kickoff.tmp")
    board.loc[keep].to_csv(temporary, index=False)
    temporary.replace(BOARD)
    print(f"[live-recent] removed {removed} started rows from the pregame board")
    return removed


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
        "provider_unavailable": int(match.eq("provider_unavailable").sum()),
        "unsupported": int(match.eq("unsupported").sum()),
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
        f"{payload['provider_unavailable']} provider unavailable, "
        f"{payload['unsupported']} unsupported"
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
        started = final_pregame_states(previous, now)
        if not started.empty:
            rows.append(started)
    candidates = read_csv_or_empty(FAVORITE_CANDIDATES)
    if not candidates.empty:
        favorite_started = favorite_handoff_states(candidates, now)
        if not favorite_started.empty:
            rows.append(favorite_started)
    market_candidates = read_csv_or_empty(MARKET_CANDIDATES)
    if not market_candidates.empty:
        market_started = final_pregame_states(market_candidates, now)
        if not market_started.empty:
            market_started["freeze_method"] = "paired_raw_market_handoff_no_classification"
            rows.append(market_started)
    if not rows:
        recovered = bootstrap_started_records(now)
        if recovered.empty:
            empty = pd.DataFrame(columns=EMPTY_COLUMNS)
            empty.to_csv(OUT, index=False)
            write_score_coverage(empty, now)
            return
        rows.append(recovered)
    live = apply_favorite_exclusions(ensure_output_columns(pd.concat(rows, ignore_index=True, sort=False)))
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
    key_columns = [column for column in ("sport", "game_id", "market_display") if column in live]
    if key_columns:
        # Within a market, a retained Favorite qualification outranks a
        # descriptive non-Favorite freeze. Among equal classifications retain
        # the earliest immutable kickoff freeze rather than a later timer pass.
        live["_favorite_priority"] = live.get(
            "red_fox_favorite", pd.Series("false", index=live.index)
        ).astype(str).str.lower().isin({"1", "true", "yes"}).astype(int)
        live = live.sort_values(
            ["_favorite_priority", "frozen_at_utc"],
            ascending=[True, False], na_position="last",
        ).drop_duplicates(key_columns, keep="last")
        live = live.drop(columns=["_favorite_priority"])
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
    # Freeze first, then remove.  This preserves the exact final published
    # pregame classification while making board expiry independent of the
    # slower anomaly rebuild completing successfully.
    expire_started_board_rows(now)
    write_score_coverage(live, now)
    print(f"[live-recent] wrote {len(live)} frozen pregame records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-only", action="store_true", help="Refresh scores without touching frozen board records")
    main(parser.parse_args().scores_only)
