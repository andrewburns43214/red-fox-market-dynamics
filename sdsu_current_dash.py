import pandas as pd

GAME = "Utah State @ San Diego State"

dash = pd.read_csv("data/dashboard.csv", dtype=str)
dash = dash[dash["game"] == GAME].copy()

dash["game_confidence"] = pd.to_numeric(dash["game_confidence"], errors="coerce")
dash["net_edge"] = pd.to_numeric(dash["net_edge"], errors="coerce")

print("\n=== SDSU CURRENT DASHBOARD ===\n")
print(dash[[
    "market_display",
    "favored_side",
    "game_confidence",
    "min_side_score",
    "max_side_score",
    "net_edge",
    "game_decision"
]].to_string(index=False))
