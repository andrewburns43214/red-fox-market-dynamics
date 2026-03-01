import pandas as pd

d = pd.read_csv("data/dashboard.csv", dtype=str)

mask = d["game"].str.contains("Mississippi State", case=False, na=False)

cols = [
    "sport","game","timing_bucket",

    "SPREAD_model_score","SPREAD_favored","SPREAD_net_edge","SPREAD_decision",
    "MONEYLINE_model_score","MONEYLINE_favored","MONEYLINE_net_edge","MONEYLINE_decision",
    "TOTAL_model_score","TOTAL_favored","TOTAL_net_edge","TOTAL_decision"
]

print(d.loc[mask, cols])
