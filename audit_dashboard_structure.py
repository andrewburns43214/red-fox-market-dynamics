import pandas as pd
d = pd.read_csv("data/dashboard.csv")
print("rows:", len(d))
print("columns:", list(d.columns))
print("has market_display:", "market_display" in d.columns)
print("has SPREAD_model_score:", any("SPREAD_model_score" in c for c in d.columns))
