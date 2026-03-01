import pandas as pd
d = pd.read_csv("data/dashboard.csv")

for col in ["SPREAD_model_score","MONEYLINE_model_score","TOTAL_model_score"]:
    if col in d.columns:
        print("\n", col)
        print(d[col].astype(float).describe())
