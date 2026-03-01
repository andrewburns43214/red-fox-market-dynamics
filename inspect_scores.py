import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

print(d[[
    "sport",
    "game",
    "SPREAD_model_score",
    "MONEYLINE_model_score",
    "TOTAL_model_score"
]].head(5))
