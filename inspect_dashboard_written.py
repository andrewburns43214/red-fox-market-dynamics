import pandas as pd
d = pd.read_csv("data/dashboard.csv")
print("ROWS", len(d), "COLS", len(d.columns))
print("HAS SPREAD_model_score", "SPREAD_model_score" in d.columns)
print("HAS MONEYLINE_model_score", "MONEYLINE_model_score" in d.columns)
print("HAS TOTAL_model_score", "TOTAL_model_score" in d.columns)
print("FIRST 40 COLS:", list(d.columns)[:40])
