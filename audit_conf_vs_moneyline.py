import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

# Load dashboard (aggregated)
dash = pd.read_csv("data/dashboard.csv")
dash["_game_time"] = pd.to_datetime(dash["_game_time"], errors="coerce")

# Filter to tonight
target_date = pd.Timestamp.now(tz="America/New_York").date()
dash = dash[dash["_game_time"].dt.date == target_date]

# Only actionable rows
dash = dash[dash["game_decision"].isin(["BET","LEAN"])]

# Load snapshots (raw odds layer)
snap = pd.read_csv("data/snapshots.csv")

# Take most recent snapshot per sport/game/market/side
snap["snapshot_ts"] = pd.to_datetime(snap["snapshot_ts"], errors="coerce")

snap = (
    snap.sort_values("snapshot_ts")
        .groupby(["sport","game_id","market_display","side"])
        .tail(1)
)

# Rename for merge clarity
snap = snap.rename(columns={
    "side": "favored_side",
    "odds": "current_odds"
})

# Merge odds into dashboard
merged = pd.merge(
    dash,
    snap[["sport","game_id","market_display","favored_side","current_odds"]],
    on=["sport","game_id","market_display","favored_side"],
    how="left"
)

# Compute implied probability (American odds)
def implied_prob(odds):
    try:
        o = float(odds)
        if o > 0:
            return 100 / (o + 100)
        else:
            return abs(o) / (abs(o) + 100)
    except:
        return np.nan

merged["implied_prob"] = merged["current_odds"].apply(implied_prob)

# Sort by confidence descending
merged = merged.sort_values("game_confidence", ascending=False)

print("\n==============================================")
print("TONIGHT — CONFIDENCE vs MONEYLINE ODDS")
print("==============================================")
print(merged.to_string(index=False))

print("\nDone.")
