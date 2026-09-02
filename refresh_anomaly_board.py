"""Build the public anomaly board without rerunning the legacy report pipeline."""

from pathlib import Path
from urllib.parse import quote

import pandas as pd

from anomaly_action_ledger import apply_recorded_signals, update_action_ledger
from anomaly_action_results import rebuild_action_results
from anomaly_board import build_anomaly_outputs, select_market_leaders
from build_live_recent import main as build_live_recent
from main import infer_market_type, normalize_side_key


DATA = Path("data")


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


def main():
    # Capture any just-started games from the prior pregame export before this
    # run replaces it. The separate file is the only source for Live & Recent.
    build_live_recent()
    snapshots = pd.read_csv(DATA / "snapshots.csv", dtype=str, keep_default_na=False)

    snapshots["market_display"] = snapshots.apply(market_for, axis=1)
    snapshots = snapshots[snapshots["market_display"].isin(["MONEYLINE", "SPREAD", "TOTAL"])].copy()
    snapshots["side_key"] = snapshots.apply(
        lambda row: normalize_side_key(row.get("sport", ""), row["market_display"], row.get("side", "")), axis=1
    )
    history = snapshots.copy()
    snapshots["timestamp"] = pd.to_datetime(snapshots["timestamp"], utc=True, errors="coerce")
    newest_snapshot = snapshots["timestamp"].max()
    active = snapshots[snapshots["timestamp"] >= newest_snapshot - pd.Timedelta(hours=2)].copy()
    latest_keys = ["sport", "game_id", "market_display", "side_key"]
    dashboard = active.sort_values("timestamp").groupby(latest_keys, as_index=False).tail(1).copy()
    dashboard["canonical_key"] = dashboard["sport"] + "|" + dashboard["game_id"]
    dashboard["_sort_time"] = dashboard.get("dk_start_iso", "")

    # This is a pregame board. Once a scheduled game has been underway for five
    # minutes, its market observations stay in history but leave the live board.
    kickoff = pd.to_datetime(dashboard.get("dk_start_iso", ""), utc=True, errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=5)
    before_expiry = len(dashboard)
    dashboard = dashboard.loc[kickoff.isna() | (kickoff > cutoff)].copy()
    print(f"[ok] kept {len(dashboard)}/{before_expiry} pregame markets after kickoff expiry")

    l2_path = DATA / "l2_consensus.csv"
    l2 = pd.read_csv(l2_path, dtype=str, keep_default_na=False) if l2_path.exists() else pd.DataFrame()
    board, events = build_anomaly_outputs(dashboard, history, l2)
    action_count = update_action_ledger(board, DATA, newest_snapshot.to_pydatetime())
    board = apply_recorded_signals(board, DATA)
    board = select_market_leaders(board)
    detail_count = write_event_detail_files(board, events)
    # Replace each public file only after its complete export is ready for Nginx.
    for frame, name in ((board, "anomaly_board.csv"), (events, "anomaly_events.csv")):
        temporary = DATA / f".{name}.tmp"
        frame.to_csv(temporary, index=False)
        temporary.replace(DATA / name)
    resolved_count = rebuild_action_results(DATA)
    print(f"[ok] wrote {len(board)} board rows, {len(events)} timeline events across {detail_count} fast detail payloads, captured {action_count} KPI candidates, and reconciled {resolved_count} results")


if __name__ == "__main__":
    main()
