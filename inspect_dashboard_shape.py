import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

print("ROWS:", len(d))
print("COLS:", len(d.columns))
print("COLUMNS:")
for c in d.columns:
    print(" -", c)

print("\nHas SPREAD_model_score:", "SPREAD_model_score" in d.columns)
print("Has MONEYLINE_model_score:", "MONEYLINE_model_score" in d.columns)
print("Has TOTAL_model_score:", "TOTAL_model_score" in d.columns)
