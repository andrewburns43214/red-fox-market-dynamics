import pandas as pd

d = pd.read_csv("data/dashboard.csv", dtype=str)

# Filter Wisconsin @ Oregon
w_o = d[d["game"].str.contains("Wisconsin @ Oregon", na=False)].copy()

# Make numeric
w_o["game_confidence"] = pd.to_numeric(w_o["game_confidence"], errors="coerce")
w_o["net_edge"] = pd.to_numeric(w_o["net_edge"], errors="coerce")

print("\n=== Wisconsin vs Oregon Confidence Scores ===\n")
print(w_o[[
    "market_display",
    "favored_side",
    "game_confidence",
    "net_edge",
    "game_decision"
]].to_string(index=False))

print("\nTotal rows:", len(w_o))
