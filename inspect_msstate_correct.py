import pandas as pd

d = pd.read_csv("data/dashboard.csv", dtype=str)

mask = d["game"].str.contains("Mississippi State", case=False, na=False)

cols = [
    "sport",
    "game",
    "market_display",
    "favored_side",
    "game_confidence",
    "min_side_score",
    "max_side_score",
    "net_edge",
    "game_decision"
]

print(d.loc[mask, cols].sort_values(["market_display"]))
