import pandas as pd
d = pd.read_csv("data/dashboard.csv")

for col in ["SPREAD_model_score","MONEYLINE_model_score","TOTAL_model_score"]:
    if col in d.columns:
        print("\n", col)
        print(">=72:", (d[col].astype(float) >= 72).sum())
        print(">=75:", (d[col].astype(float) >= 75).sum())
