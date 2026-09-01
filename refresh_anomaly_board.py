"""Build the public anomaly board without rerunning the legacy report pipeline."""

from pathlib import Path

import pandas as pd

from anomaly_action_ledger import apply_recorded_signals, update_action_ledger
from anomaly_action_results import rebuild_action_results
from anomaly_board import build_anomaly_outputs
from main import infer_market_type, normalize_side_key


DATA = Path("data")


def market_for(row):
    return infer_market_type(row.get("side", ""), row.get("current_line", ""))


def main():
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

    l2_path = DATA / "l2_consensus.csv"
    l2 = pd.read_csv(l2_path, dtype=str, keep_default_na=False) if l2_path.exists() else pd.DataFrame()
    board, events = build_anomaly_outputs(dashboard, history, l2)
    action_count = update_action_ledger(board, DATA, newest_snapshot.to_pydatetime())
    board = apply_recorded_signals(board, DATA)
    # Replace each public file only after its complete export is ready for Nginx.
    for frame, name in ((board, "anomaly_board.csv"), (events, "anomaly_events.csv")):
        temporary = DATA / f".{name}.tmp"
        frame.to_csv(temporary, index=False)
        temporary.replace(DATA / name)
    resolved_count = rebuild_action_results(DATA)
    print(f"[ok] wrote {len(board)} board rows, {len(events)} timeline events, captured {action_count} KPI candidates, and reconciled {resolved_count} results")


if __name__ == "__main__":
    main()
