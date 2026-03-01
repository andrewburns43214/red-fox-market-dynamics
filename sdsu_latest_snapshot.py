import pandas as pd

GAME = "Utah State @ San Diego State"

snap = pd.read_csv("data/snapshots.csv", dtype=str)
s = snap[snap["game"] == GAME].copy()

s = s.sort_values("timestamp")

print("\n=== SDSU LATEST SNAPSHOT ROWS ===\n")
print(s[[
    "timestamp",
    "market",
    "side",
    "bets_pct",
    "money_pct",
    "open_line",
    "current_line"
]].tail(12).to_string(index=False))
