import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

# Load dashboard
d = pd.read_csv("data/dashboard.csv")

# Parse time
d["_game_time"] = pd.to_datetime(d["_game_time"], errors="coerce")

# Filter to tonight
target_date = pd.Timestamp.now(tz="America/New_York").date()
d = d[d["_game_time"].dt.date == target_date]

# Remove NHL
d = d[d["sport"] != "nhl"]

# Keep only ML + Spread
d = d[d["market_display"].isin(["MONEYLINE","SPREAD"])]

# Sort by confidence descending
d = d.sort_values("game_confidence", ascending=False)

print("\n===== TONIGHT ML + SPREAD (NON-HOCKEY) =====\n")
print(d.to_string(index=False))
print("\nTotal rows:", len(d))
