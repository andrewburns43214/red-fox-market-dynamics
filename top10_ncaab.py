import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

d = pd.read_csv("data/dashboard.csv")

# Parse time
d["_game_time"] = pd.to_datetime(d["_game_time"], errors="coerce")

# Filter to tonight (Eastern time assumption already baked into dashboard)
target_date = pd.Timestamp.now(tz="America/New_York").date()
d = d[d["_game_time"].dt.date == target_date]

# Filter to NCAAB only
d = d[d["sport"] == "ncaab"]

# Sort by confidence
d = d.sort_values("game_confidence", ascending=False)

# Top 10
top10 = d.head(10)

print("\n===== TOP 10 NCAAB TONIGHT =====\n")
print(top10[[
    "game",
    "market_display",
    "favored_side",
    "game_confidence",
    "net_edge",
    "game_decision",
    "_game_time"
]].to_string(index=False))

print("\nDone.")
