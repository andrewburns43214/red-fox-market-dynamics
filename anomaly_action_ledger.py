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


def apply_recorded_signals(board, data_dir):
    """Keep a qualified signal visible after its live split later changes."""
    if board is None or board.empty:
        return board
    board = board.copy()
    for column in ("recorded_reaction", "recorded_action_type", "recorded_action_side", "recorded_action_line", "recorded_at", "recorded_note"):
        board[column] = ""
    path = Path(data_dir) / "anomaly_action_ledger.csv"
    if not path.exists():
        return board
    ledger = pd.read_csv(path, dtype=str, keep_default_na=False)
    needed = {"sport", "game_id", "market_display", "observed_side", "reaction", "action_type", "action_side", "action_line", "first_anomaly_seen", "captured_at_utc"}
    if ledger.empty or not needed.issubset(ledger.columns):
        return board
    ledger = ledger.sort_values("captured_at_utc").drop_duplicates(["sport", "game_id", "market_display", "observed_side"], keep="last")
    lookup = ledger.set_index(["sport", "game_id", "market_display", "observed_side"])
    for index, row in board.iterrows():
        if str(row.get("reaction", "")) != "Watch":
            continue
        key = (str(row.get("sport", "")), str(row.get("game_id", "")), str(row.get("market_display", "")), str(row.get("flagged_side", "")))
        if key not in lookup.index:
            continue
        saved = lookup.loc[key]
        reaction = str(saved["reaction"])
        board.at[index, "recorded_reaction"] = reaction
        board.at[index, "recorded_action_type"] = saved["action_type"]
        board.at[index, "recorded_action_side"] = saved["action_side"]
        board.at[index, "recorded_action_line"] = saved["action_line"]
        board.at[index, "recorded_at"] = saved["first_anomaly_seen"]
        board.at[index, "recorded_note"] = f"Recorded {reaction} at {saved['action_line']}; the current snapshot changed after the trigger."
        board.at[index, "anomaly_sort"] = min(float(row.get("anomaly_sort", 99)), 3.0 if reaction == "Contrarian" else 5.0)
    board = board.sort_values(["anomaly_sort", "board_rank"], kind="mergesort").reset_index(drop=True)
    board["board_rank"] = range(1, len(board) + 1)
    return board
