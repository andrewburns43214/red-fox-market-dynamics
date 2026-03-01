import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

d = pd.read_csv("data/dashboard.csv")

# Parse time
d["_game_time"] = pd.to_datetime(d["_game_time"], errors="coerce")

# Filter to tonight (Eastern)
target_date = pd.Timestamp.now(tz="America/New_York").date()
d = d[d["_game_time"].dt.date == target_date]

if d.empty:
    print("No games found for tonight.")
    exit()

# Make sure odds column exists
odds_cols = [c for c in d.columns if "odds" in c.lower()]
print("\nDetected odds columns:", odds_cols)

# Keep only actionable rows
d = d[d["game_decision"].isin(["BET","LEAN"])]

# Sort by confidence descending
d = d.sort_values("game_confidence", ascending=False)

print("\n======================================")
print("TONIGHT — CONFIDENCE vs ODDS (ALL COLS)")
print("======================================")

print(d.to_string(index=False))

print("\nDone.")
