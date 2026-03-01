import pandas as pd

# Load dashboard
d = pd.read_csv("data/dashboard.csv", dtype=str)

# Filter NCAAB
d = d[d["sport"] == "ncaab"].copy()

# Convert numeric fields
d["game_confidence"] = pd.to_numeric(d["game_confidence"], errors="coerce")
d["net_edge"] = pd.to_numeric(d["net_edge"], errors="coerce")

# Parse game time
if "_game_time" not in d.columns:
    raise SystemExit("No _game_time column in dashboard")

d["_dt"] = pd.to_datetime(d["_game_time"], errors="coerce")

# Filter by 02/25 date
d = d[d["_dt"].dt.strftime("%m/%d") == "02/25"]

# Sort by confidence descending
d = d.sort_values("game_confidence", ascending=False)

print("\n=== NCAAB – 02/25 (All Markets) ===\n")
print(d[[
    "game",
    "market_display",
    "favored_side",
    "game_confidence",
    "net_edge",
    "game_decision"
]].to_string(index=False))

print("\nTotal rows:", len(d))
