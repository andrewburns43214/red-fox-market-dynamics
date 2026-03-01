import pandas as pd
from datetime import datetime
import pytz

# Load dashboard
d = pd.read_csv("data/dashboard.csv", dtype=str)

# Filter NCAAB
d = d[d["sport"] == "ncaab"].copy()

# Convert game time if present
if "_game_time" in d.columns:
    d["_dt"] = pd.to_datetime(d["_game_time"], errors="coerce")
else:
    d["_dt"] = None

# Sort by confidence descending
d["game_confidence"] = pd.to_numeric(d["game_confidence"], errors="coerce")
d["net_edge"] = pd.to_numeric(d["net_edge"], errors="coerce")

d = d.sort_values(["game_confidence"], ascending=False)

cols = [
    "game",
    "market_display",
    "favored_side",
    "game_confidence",
    "net_edge",
    "game_decision"
]

print("\n=== NCAAB – Current Window ===\n")
print(d[cols].to_string(index=False))
print("\nTotal rows:", len(d))
