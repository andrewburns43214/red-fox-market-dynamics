import pandas as pd
import numpy as np
import re

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

# ---------------------------
# Load dashboard
# ---------------------------
dash = pd.read_csv("data/dashboard.csv")
dash["_game_time"] = pd.to_datetime(dash["_game_time"], errors="coerce")

target_date = pd.Timestamp.now(tz="America/New_York").date()
dash = dash[dash["_game_time"].dt.date == target_date]
dash = dash[dash["game_decision"].isin(["BET","LEAN"])]

# ---------------------------
# Load snapshots
# ---------------------------
snap = pd.read_csv("data/snapshots.csv")
snap["timestamp"] = pd.to_datetime(snap["timestamp"], errors="coerce")

# Keep latest per sport/game/side/market
snap = (
    snap.sort_values("timestamp")
        .groupby(["sport","game_id","market","side"])
        .tail(1)
)

# Extract American odds from current_line
def extract_odds(line):
    if isinstance(line, str):
        m = re.search(r'@ ([+-]?\d+)', line)
        if m:
            return int(m.group(1))
    return np.nan

snap["current_odds"] = snap["current_line"].apply(extract_odds)

# Align naming
snap = snap.rename(columns={
    "market": "market_display",
    "side": "favored_side"
})

# ---------------------------
# Merge
# ---------------------------
merged = pd.merge(
    dash,
    snap[["sport","game_id","market_display","favored_side","current_odds"]],
    on=["sport","game_id","market_display","favored_side"],
    how="left"
)

# Compute implied probability
def implied_prob(odds):
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

merged["implied_prob"] = merged["current_odds"].apply(implied_prob)

merged = merged.sort_values("game_confidence", ascending=False)

print("\n==============================================")
print("TONIGHT — CONFIDENCE vs CURRENT ODDS")
print("==============================================")
print(merged.to_string(index=False))

print("\nDone.")
