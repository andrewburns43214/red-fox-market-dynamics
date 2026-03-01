import pandas as pd

GAME = "Utah State @ San Diego State"

d = pd.read_csv("data/dashboard.csv", dtype=str)
d = d[d["game"] == GAME].copy()

d["game_confidence"] = pd.to_numeric(d["game_confidence"], errors="coerce")
d["net_edge"] = pd.to_numeric(d["net_edge"], errors="coerce")

print("\n=== Aggregated Dashboard Rows ===\n")
print(d[[
    "market_display",
    "favored_side",
    "game_confidence",
    "min_side_score",
    "max_side_score",
    "net_edge",
    "game_decision"
]].to_string(index=False))
