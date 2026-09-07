"""Build the public anomaly board without rerunning the legacy report pipeline."""

import json
import os
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd

from anomaly_action_ledger import apply_recorded_signals, update_action_ledger
from anomaly_action_results import rebuild_action_results
from anomaly_board import build_anomaly_outputs, select_market_leaders
from build_live_recent import main as build_live_recent
from cross_market_integrity import CROSS_MARKET_COLUMNS, apply_cross_market_integrity
from cross_market_split import CROSS_MARKET_SPLIT_COLUMNS, apply_cross_market_split
from main import infer_market_type, normalize_side_key
from red_fox_favorite import FAVORITE_COLUMNS, apply_red_fox_favorites, update_favorite_tracking


DATA = Path(os.environ.get("REDFOX_DATA_DIR", "data"))
PUBLIC_TIMEZONE = ZoneInfo("America/New_York")
FOOTBALL_SPORTS = {"nfl", "ncaaf", "cfb"}
# The 2026 opening slate is available in the source before the first Tuesday
# board rollover.  This bounded exception publishes only that slate; after
# Sep. 14 the normal Tuesday-to-Monday NFL window applies without exception.
NFL_OPENING_WEEK_EXCEPTION_ACTIVE_START = pd.Timestamp("2026-09-01", tz=PUBLIC_TIMEZONE)
NFL_OPENING_WEEK_START = pd.Timestamp("2026-09-09", tz=PUBLIC_TIMEZONE)
NFL_OPENING_WEEK_END_EXCLUSIVE = pd.Timestamp("2026-09-15", tz=PUBLIC_TIMEZONE)
PUBLIC_EXPORT_COLUMNS = {
    "anomaly_board.csv": [
        "sport", "game_id", "canonical_key", "kickoff_time", "kickoff_sort", "kickoff_iso", "game", "market_display",
        "flagged_side", "focus_basis", "action_side", "action_line", "action_type", "action_basis", "kpi_eligible",
        "reaction", "path", "context_chips", "anomaly_chips", "bets_pct", "money_pct", "open_line", "current_line",
        "path_summary", "reason", "data_badge", "observation_count", "first_anomaly_seen", "max_excursion",
        "return_toward_open", "broader_market_comparison", "key_number_note", "key_numbers_crossed", "open_line_value",
        "current_line_value", "move_abs", "line_move_abs", "price_move_pct", "line_response", "price_response_pct",
        "response_direction", "whipsaw_recovered", "active_worsening_reversal", "key_number_pinned", "evidence_role", "evidence_polarity",
        "movement_unit", "line_dir_changes",
        "path_min", "path_max", "observed_path", "rank_reason", "anomaly_sort", "maturity_sort", "severity_sort",
        "board_rank", "recorded_reaction", "recorded_action_type", "recorded_action_side", "recorded_action_line",
        "recorded_at", "recorded_note", "market_sides", "read_anchor_side", "supported_side", "directional_lean_side", "market_rationale",
        *CROSS_MARKET_COLUMNS, *CROSS_MARKET_SPLIT_COLUMNS, *FAVORITE_COLUMNS,
    ],
    "anomaly_events.csv": [
        "sport", "game_id", "canonical_key", "game", "market_display", "flagged_side", "focus_basis", "action_side",
        "action_line", "action_type", "action_basis", "kpi_eligible", "timestamp", "step_index", "observation_count",
        "line_value", "line_display", "price_odds", "implied_pct", "bets_pct", "money_pct", "is_open", "is_current",
        "reaction", "path", "first_anomaly_seen", "max_excursion", "return_toward_open", "broader_market_comparison",
        "key_number_note", "key_numbers_crossed",
    ],
}


def filter_fresh_market_rows(dashboard, now=None, max_age_minutes=None):
    """Do not publish an old split state as if it were a live market.

    A failed sport scrape can leave otherwise valid paired rows in the two-hour
    working history.  Keeping those rows on the public board is worse than
    omitting them: customers cannot distinguish them from a current capture.
    The threshold is configurable for operational incidents, while the normal
    value leaves ample room for a complete sequential multi-sport pass.
    """
    if dashboard is None or dashboard.empty or "timestamp" not in dashboard.columns:
        return pd.DataFrame() if dashboard is None else dashboard.copy()
    if max_age_minutes is None:
        max_age_minutes = int(os.environ.get("REDFOX_PUBLIC_MAX_MARKET_AGE_MINUTES", "10"))
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    captured = pd.to_datetime(dashboard["timestamp"], utc=True, errors="coerce")
    return dashboard.loc[captured >= current - pd.Timedelta(minutes=max_age_minutes)].copy()


def write_board_freshness(dashboard, data_dir=DATA, now=None):
    """Atomically record the real source age of the just-published board."""
    if dashboard is None or dashboard.empty or "timestamp" not in dashboard.columns:
        path = data_dir / "freshness.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError):
            payload = {}
        payload.update(dk_ts=None, board_oldest_ts=None, board_newest_ts=None,
                       board_market_count=0, board_published_at=str(now or pd.Timestamp.now(tz="UTC")))
        data_dir.mkdir(parents=True, exist_ok=True)
        temporary = data_dir / ".freshness.json.tmp"
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        return None
    captured = pd.to_datetime(dashboard["timestamp"], utc=True, errors="coerce").dropna()
    if captured.empty:
        return None
    oldest, newest = captured.min(), captured.max()
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    path = data_dir / "freshness.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError, TypeError):
        payload = {}
    market_count = 0
    keys = [column for column in ("sport", "game_id", "market_display") if column in dashboard.columns]
    if keys:
        market_count = int(dashboard.loc[:, keys].drop_duplicates().shape[0])
    payload.update({
        # dk_ts remains for backward compatibility, but now means the oldest
        # source capture a customer can see, never the runner wall clock.
        "dk_ts": oldest.isoformat(),
        "board_oldest_ts": oldest.isoformat(),
        "board_newest_ts": newest.isoformat(),
        "board_market_count": market_count,
        "board_published_at": current.isoformat(),
    })
    temporary = data_dir / ".freshness.json.tmp"
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return oldest, newest, market_count


def load_current_l2(data_dir, as_of):
    """Load L2 only while its capture can still describe the current board.

    ``l2_consensus.csv`` is a retained, append-only historical source.  Parsing
    an old archive on every ten-minute board pass provides no current-market
    context and previously cost a full CSV parse even when it had no active
    keys.  Keep the archive intact, but treat a capture older than the
    configurable window as unavailable for the live board.  This is the same
    result the downstream active-key filter produced for stale data.
    """
    path = Path(data_dir) / "l2_consensus.csv"
    if not path.exists():
        return pd.DataFrame()
    max_age_hours = float(os.environ.get("REDFOX_L2_MAX_AGE_HOURS", "24"))
    reference = pd.Timestamp(as_of)
    reference = reference.tz_localize("UTC") if reference.tzinfo is None else reference.tz_convert("UTC")
    modified = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    if modified < reference - pd.Timedelta(hours=max_age_hours):
        print(f"[ok] skipped stale L2 raw archive ({modified.isoformat()}); retained for history")
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def filter_publication_eligible_markets(dashboard, now=None):
    """Apply public football windows before board ranking/export.

    Non-football rows retain their existing publication behavior.  CFB keeps
    its existing rolling eight-calendar-day window.  NFL uses a Tuesday-to-
    Monday board week so the current week remains visible through Monday Night
    Football and Week +2 source data is never published early.
    """
    if dashboard is None or dashboard.empty:
        return pd.DataFrame() if dashboard is None else dashboard.copy()
    work = dashboard.copy()
    now = pd.Timestamp.now(tz=PUBLIC_TIMEZONE) if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize(PUBLIC_TIMEZONE)
    else:
        now = now.tz_convert(PUBLIC_TIMEZONE)
    start = now.normalize()
    end_exclusive = start + pd.DateOffset(days=8)
    kickoff = pd.to_datetime(work.get("dk_start_iso", ""), errors="coerce", utc=True).dt.tz_convert(PUBLIC_TIMEZONE)
    sport = work.get("sport", "").fillna("").astype(str).str.strip().str.lower()
    football = sport.isin(FOOTBALL_SPORTS)
    nfl = sport.eq("nfl")

    # Opening-week exception: publish exactly Sep. 9 through Sep. 14 before
    # the first regular Tuesday rollover, never the following NFL week.
    if NFL_OPENING_WEEK_EXCEPTION_ACTIVE_START <= now < NFL_OPENING_WEEK_END_EXCLUSIVE:
        nfl_start = NFL_OPENING_WEEK_START
        nfl_end_exclusive = NFL_OPENING_WEEK_END_EXCLUSIVE
    else:
        # pandas weekday is Monday=0; Tuesday is the first day of this window.
        nfl_start = start - pd.DateOffset(days=(now.weekday() - 1) % 7)
        nfl_end_exclusive = nfl_start + pd.DateOffset(days=7)

    eligible_other_football = kickoff.notna() & (kickoff >= start) & (kickoff < end_exclusive)
    eligible_nfl = kickoff.notna() & (kickoff >= nfl_start) & (kickoff < nfl_end_exclusive)
    eligible = ~football | ((~nfl) & eligible_other_football) | (nfl & eligible_nfl)
    return work.loc[eligible].copy()


def _event_detail_filename(sport, game_id):
    return f"{quote(str(sport).strip().lower(), safe='')}--{quote(str(game_id).strip(), safe='')}.json"


def write_event_detail_files(board, events, details_dir=DATA / "anomaly_event_details"):
    """Publish small, game-scoped timeline payloads for instant detail views."""
    required = {"sport", "game_id"}
    if board.empty or events.empty or not required.issubset(board.columns) or not required.issubset(events.columns):
        return 0

    details_dir.mkdir(parents=True, exist_ok=True)
    key_frames = [board.loc[:, ["sport", "game_id"]].copy()]
    # Live & Recent entries can outlive the pregame board briefly. Include
    # their timelines too so a detail click never falls back to the full archive.
    live_recent_path = DATA / "live_recent.csv"
    if live_recent_path.exists():
        try:
            live_recent = pd.read_csv(live_recent_path, dtype=str, keep_default_na=False)
            if required.issubset(live_recent.columns):
                key_frames.append(live_recent.loc[:, ["sport", "game_id"]])
        except (OSError, pd.errors.ParserError):
            pass
    active_keys = pd.concat(key_frames, ignore_index=True)
    active_keys["sport"] = active_keys["sport"].fillna("").astype(str).str.strip().str.lower()
    active_keys["game_id"] = active_keys["game_id"].fillna("").astype(str).str.strip()
    active_keys = active_keys[(active_keys["sport"] != "") & (active_keys["game_id"] != "")].drop_duplicates()

    scoped = events.copy()
    scoped["sport"] = scoped["sport"].fillna("").astype(str).str.strip().str.lower()
    scoped["game_id"] = scoped["game_id"].fillna("").astype(str).str.strip()
    scoped = scoped.merge(active_keys, on=["sport", "game_id"], how="inner")

    written = 0
    for (sport, game_id), frame in scoped.groupby(["sport", "game_id"], sort=False):
        target = details_dir / _event_detail_filename(sport, game_id)
        temporary = details_dir / f".{target.name}.tmp"
        temporary.write_text(
            frame.where(pd.notna(frame), "").to_json(orient="records", date_format="iso"),
            encoding="utf-8",
        )
        temporary.replace(target)
        written += 1
    return written


def market_for(row):
    return infer_market_type(row.get("side", ""), row.get("current_line", ""))


def markets_for(frame):
    """Vectorized equivalent of ``market_for`` for the lifetime archive."""
    if frame is None or frame.empty:
        return pd.Series(dtype=str, index=getattr(frame, "index", None))
    side = frame.get("side", pd.Series("", index=frame.index)).fillna("").astype(str)
    line = frame.get("current_line", pd.Series("", index=frame.index)).fillna("").astype(str)
    normalized_side = side.str.strip().str.lower()
    result = pd.Series("", index=frame.index, dtype=str)
    total = normalized_side.str.startswith(("over", "under"))
    result.loc[total] = "TOTAL"
    spread = ~total & side.str.contains(r"[+-]\d+(?:\.\d+)?\b", regex=True, na=False)
    result.loc[spread] = "SPREAD"
    price = line.str.strip().str.replace("?", "-", regex=False).str.extract(r"@\s*([+-]\d{3,7})\s*$", expand=False)
    price_number = pd.to_numeric(price, errors="coerce").abs()
    moneyline = ~total & ~spread & price_number.between(100, 1_000_000, inclusive="both")
    result.loc[moneyline] = "MONEYLINE"
    return result


def latest_synchronized_market_rows(active):
    """Return only the latest complete, same-timestamp two-side market states.

    Selecting the latest row separately for each side can combine an Over from
    one scrape with an Under from a later scrape.  That is not a real market
    state and can create impossible board rows.  The public board therefore
    advances a market only when both opposing sides were observed in the same
    source snapshot timestamp.
    """
    if active is None or active.empty:
        return pd.DataFrame() if active is None else active.copy()

    keys = ["sport", "game_id", "market_display"]
    required = [*keys, "side_key", "timestamp"]
    if any(column not in active.columns for column in required):
        return active.iloc[0:0].copy()

    work = active.dropna(subset=["timestamp"]).copy()
    work["side_key"] = work["side_key"].fillna("").astype(str).str.strip()
    work = work[work["side_key"] != ""].copy()
    if work.empty:
        return work

    # A snapshot must contain both distinct market sides.  Keep the newest
    # complete timestamp per market, then retain one final capture per side in
    # the unlikely event a scraper retry duplicated a row within that snapshot.
    counts = (
        work.groupby([*keys, "timestamp"], dropna=False)["side_key"]
        .nunique()
        .reset_index(name="side_count")
    )
    complete = counts[counts["side_count"] >= 2]
    if complete.empty:
        return work.iloc[0:0].copy()
    latest = (
        complete.sort_values("timestamp", kind="mergesort")
        .groupby(keys, as_index=False, sort=False)
        .tail(1)
        .loc[:, [*keys, "timestamp"]]
    )
    aligned = work.merge(latest, on=[*keys, "timestamp"], how="inner")
    return aligned.drop_duplicates([*keys, "side_key"], keep="last").copy()


def complete_public_market_rows(dashboard):
    """Keep only paired markets with the essentials a customer can inspect."""
    if dashboard is None or dashboard.empty:
        return pd.DataFrame() if dashboard is None else dashboard.copy()
    keys = ["sport", "game_id", "market_display"]
    required = [*keys, "side_key", "side", "bets_pct", "money_pct", "open_line", "current_line", "dk_start_iso"]
    if any(column not in dashboard.columns for column in required):
        return dashboard.iloc[0:0].copy()
    work = dashboard.copy()
    complete_row = pd.Series(True, index=work.index)
    for column in ["side_key", "side", "bets_pct", "money_pct", "open_line", "current_line", "dk_start_iso"]:
        complete_row &= work[column].fillna("").astype(str).str.strip().ne("")
    summary = (
        work.assign(_complete=complete_row)
        .groupby(keys, dropna=False)
        .agg(side_count=("side_key", "nunique"), complete_count=("_complete", "sum"))
        .reset_index()
    )
    eligible = summary[(summary["side_count"] == 2) & (summary["complete_count"] == 2)][keys]
    return work.merge(eligible, on=keys, how="inner")


def main():
    from publication_coverage import PublicationCoverage
    coverage = PublicationCoverage(DATA, pd.Timestamp.now(tz="UTC"))
    try:
        _refresh(coverage)
    except BaseException as error:
        coverage.store.finish(coverage.run_id, "PUBLICATION_FAILED", type(error).__name__ + ": " + str(error))
        raise


def _refresh(coverage):
    DATA.mkdir(parents=True, exist_ok=True)
    # Capture any just-started games from the prior pregame export before this
    # run replaces it. The separate file is the only source for Live & Recent.
    if DATA == Path("data"):
        build_live_recent()
    snapshot_path = DATA / "snapshots.csv"
    snapshots = pd.read_csv(snapshot_path, dtype=str, keep_default_na=False) if snapshot_path.exists() else pd.DataFrame(
        columns=["sport", "game_id", "game", "side", "current_line", "open_line", "bets_pct", "money_pct", "dk_start_iso", "timestamp"])

    snapshots["market_display"] = markets_for(snapshots)
    coverage.seed(snapshots)
    supported = snapshots[snapshots["market_display"].isin(["MONEYLINE", "SPREAD", "TOTAL"])].copy()
    coverage.stage(snapshots, supported, "NORMALIZATION_FAILED")
    snapshots = coverage.validated(supported)
    snapshots["timestamp"] = pd.to_datetime(snapshots["timestamp"], utc=True, errors="coerce")
    coverage.stage(snapshots, snapshots[snapshots["timestamp"].notna()], "CAPTURE_TIMESTAMP_INVALID")
    newest_snapshot = snapshots["timestamp"].max()
    if pd.isna(newest_snapshot):
        newest_snapshot = coverage.now
    active = snapshots[snapshots["timestamp"] >= newest_snapshot - pd.Timedelta(hours=2)].copy()
    coverage.stage(snapshots, active, "STALE_CAPTURE")
    active["side_key"] = active.apply(
        lambda row: normalize_side_key(row.get("sport", ""), row["market_display"], row.get("side", "")), axis=1
    ) if len(active) else pd.Series(dtype=str)
    dashboard = latest_synchronized_market_rows(active)
    coverage.stage(active, dashboard, "AWAITING_COMPLETE_PAIR")
    coverage.pairs(dashboard)
    complete = complete_public_market_rows(dashboard)
    coverage.stage(dashboard, complete, "INCOMPLETE_MARKET_DATA")
    dashboard = complete
    # Only currently represented markets need their retained lifetime path.
    # Filtering first avoids normalizing every side in unrelated old games on
    # each live publish while preserving the full opener/timeline for the board.
    history_keys = dashboard[["sport", "game_id", "market_display"]].drop_duplicates()
    history = snapshots.merge(history_keys, on=["sport", "game_id", "market_display"], how="inner")
    history["side_key"] = history.apply(
        lambda row: normalize_side_key(row.get("sport", ""), row["market_display"], row.get("side", "")), axis=1
    ) if len(history) else pd.Series(dtype=str)
    complete_market_count = dashboard[["sport", "game_id", "market_display"]].drop_duplicates().shape[0]
    print(f"[ok] kept {complete_market_count} complete customer-inspectable same-snapshot markets")
    before_freshness = len(dashboard)
    fresh = filter_fresh_market_rows(dashboard, now=coverage.now)
    coverage.stage(dashboard, fresh, "STALE_CAPTURE")
    dashboard = fresh
    print(f"[ok] kept {len(dashboard)}/{before_freshness} rows within the public source-freshness window")
    dashboard["canonical_key"] = dashboard["sport"] + "|" + dashboard["game_id"]
    dashboard["_sort_time"] = dashboard.get("dk_start_iso", "")

    # This is a customer-facing pregame board. A market must have a scheduled
    # kickoff, and once it has been underway for five minutes its observations
    # stay in history but leave the live board.
    kickoff = pd.to_datetime(dashboard.get("dk_start_iso", ""), utc=True, errors="coerce")
    cutoff = coverage.now - pd.Timedelta(minutes=5)
    before_expiry = len(dashboard)
    pregame = dashboard.loc[kickoff.notna() & (kickoff > cutoff)].copy()
    coverage.stage(dashboard, pregame, "KICKOFF_GATE")
    dashboard = pregame
    print(f"[ok] kept {len(dashboard)}/{before_expiry} pregame markets after kickoff expiry")
    before_window = len(dashboard)
    in_window = filter_publication_eligible_markets(dashboard, now=coverage.now)
    coverage.stage(dashboard, in_window, "OUTSIDE_PUBLICATION_WINDOW")
    dashboard = in_window
    print(f"[ok] kept {len(dashboard)}/{before_window} markets after rolling football publication window")

    l2 = load_current_l2(DATA, newest_snapshot)
    # Evaluate timing against the latest source capture, not the web server's
    # wall clock. This keeps Late deterministic and prevents historical data
    # or a delayed refresh from being mislabeled as a closing-window move.
    board, events = build_anomaly_outputs(dashboard, history, l2, as_of=newest_snapshot.to_pydatetime())
    # The evaluator's existing history requirement is unchanged. Every input
    # market omitted by it receives an explicit terminal state.
    coverage.stage(dashboard, board, "INSUFFICIENT_HISTORY")
    from publication_coverage import keys as coverage_keys
    coverage.gate_ready = coverage_keys(board)
    action_count = update_action_ledger(board, DATA, newest_snapshot.to_pydatetime())
    board = apply_recorded_signals(board, DATA)
    board = select_market_leaders(board)
    board = apply_cross_market_integrity(board, history, as_of=newest_snapshot)
    board = apply_cross_market_split(board)
    board = apply_red_fox_favorites(board, as_of=newest_snapshot)
    board = update_favorite_tracking(board, DATA, as_of=newest_snapshot)
    detail_count = write_event_detail_files(board, events, details_dir=DATA / "anomaly_event_details")
    # Replace each public file only after its complete export is ready for Nginx.
    for frame, name in ((board, "anomaly_board.csv"), (events, "anomaly_events.csv")):
        temporary = DATA / f".{name}.tmp"
        # A valid, header-only CSV keeps the browser export and downstream
        # readers parseable when the existing kickoff/publication gates leave
        # no currently eligible markets.
        output = frame if len(frame.columns) else pd.DataFrame(columns=PUBLIC_EXPORT_COLUMNS[name])
        output.to_csv(temporary, index=False)
        temporary.replace(DATA / name)
    coverage_summary = coverage.publish(board, DATA / "anomaly_board.csv", filter_publication_eligible_markets)
    print("[coverage] " + json.dumps(coverage_summary["sports"], sort_keys=True))
    # ``coverage.now`` is the deterministic evaluation clock captured at run
    # start.  Customer-facing publish age must record when the completed files
    # actually became available, especially when a large board takes minutes.
    freshness = write_board_freshness(dashboard, data_dir=DATA)
    resolved_count = rebuild_action_results(DATA)
    freshness_summary = "no current source rows" if freshness is None else (
        f"source range {freshness[0].isoformat()} to {freshness[1].isoformat()} across {freshness[2]} markets"
    )
    print(f"[ok] wrote {len(board)} board rows, {len(events)} timeline events across {detail_count} fast detail payloads, captured {action_count} KPI candidates, reconciled {resolved_count} results, and published {freshness_summary}")


if __name__ == "__main__":
    main()
