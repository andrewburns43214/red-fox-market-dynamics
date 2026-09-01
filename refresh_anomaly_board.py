"""Build the public anomaly board without rerunning the legacy report pipeline."""

from pathlib import Path

import pandas as pd

from anomaly_board import build_anomaly_outputs
from main import infer_market_type, normalize_side_key


DATA = Path("data")


def market_for(row):
    return infer_market_type(row.get("side", ""), row.get("current_line", ""))


def main():
    dashboard = pd.read_csv(DATA / "dashboard.csv", dtype=str, keep_default_na=False)
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
    board.to_csv(DATA / "anomaly_board.csv", index=False)
    events.to_csv(DATA / "anomaly_events.csv", index=False)
    print(f"[ok] wrote {len(board)} board rows and {len(events)} timeline events")


if __name__ == "__main__":
    main()
