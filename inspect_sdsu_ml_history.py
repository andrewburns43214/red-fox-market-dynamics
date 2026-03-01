import pandas as pd

GAME = "Utah State @ San Diego State"
MARKET = "MONEYLINE"

snap = pd.read_csv("data/snapshots.csv", dtype=str)

s = snap[(snap["game"] == GAME) & (snap["market"] == MARKET)].copy()

if s.empty:
    print("No snapshot rows found.")
    raise SystemExit()

# Convert numerics
for col in ["bets_pct","money_pct","open_line","current_line"]:
    if col in s.columns:
        s[col] = pd.to_numeric(s[col], errors="coerce")

# Sort by timestamp
s = s.sort_values("timestamp")

print("\n=== SDSU MONEYLINE SNAPSHOT HISTORY ===\n")
print(s[[
    "timestamp",
    "side",
    "bets_pct",
    "money_pct",
    "open_line",
    "current_line"
]].to_string(index=False))
