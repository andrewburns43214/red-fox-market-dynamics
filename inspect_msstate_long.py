import pandas as pd

d = pd.read_csv("data/dashboard.csv", dtype=str)

mask = d["game"].str.contains("Mississippi State", case=False, na=False)

cols = [
    "sport",
    "game",
    "market_display",
    "favored_side",
    "model_score",
    "net_edge",
    "decision",
    "timing_bucket"
]

print(d.loc[mask, cols].sort_values(["market_display"]))
