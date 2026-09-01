from pathlib import Path

import pandas as pd


ACTION_LEDGER_COLUMNS = [
    "action_id", "captured_at_utc", "first_anomaly_seen", "sport", "game_id", "canonical_key",
    "game", "market_display", "reaction", "path", "observed_side", "observed_line",
    "action_side", "action_line", "action_type", "action_basis", "bets_pct", "money_pct",
    "observation_count", "board_rank",
]


def update_action_ledger(board, data_dir, captured_at):
    """Append each KPI-eligible decision once, preserving its contemporaneous line."""
    if board is None or board.empty or "kpi_eligible" not in board.columns:
        return 0

    candidates = board[board["kpi_eligible"].astype(str).str.lower().eq("true")].copy()
    if candidates.empty:
        return 0

    data_dir = Path(data_dir)
    ledger_path = data_dir / "anomaly_action_ledger.csv"
    candidates["action_id"] = candidates.apply(
        lambda row: "|".join(str(row.get(column, "")) for column in (
            "sport", "game_id", "market_display", "action_type", "action_side", "first_anomaly_seen",
        )),
        axis=1,
    )
    candidates["captured_at_utc"] = captured_at.isoformat()
    candidates = candidates.rename(columns={"flagged_side": "observed_side", "current_line": "observed_line"})
    for column in ACTION_LEDGER_COLUMNS:
        if column not in candidates.columns:
            candidates[column] = ""
    candidates = candidates[ACTION_LEDGER_COLUMNS]

    existing_ids = set()
    if ledger_path.exists():
        existing = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
        for column in ACTION_LEDGER_COLUMNS:
            if column not in existing.columns:
                existing[column] = ""
        existing_ids = set(existing["action_id"].astype(str))
        combined = pd.concat([existing[ACTION_LEDGER_COLUMNS], candidates], ignore_index=True)
    else:
        combined = candidates
    combined = combined.drop_duplicates(subset=["action_id"], keep="first")
    temporary = data_dir / ".anomaly_action_ledger.csv.tmp"
    combined.to_csv(temporary, index=False)
    temporary.replace(ledger_path)
    return sum(action_id not in existing_ids for action_id in candidates["action_id"])
