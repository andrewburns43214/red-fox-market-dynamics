import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

print("ROWS:", len(d))
print("\nCOLUMNS:")
for c in d.columns:
    print("-", c)

print("\nSample values used by metrics:\n")

cols = [
    "sport",
    "game_id",
    "market_display",
    "favored_side",
    "game_confidence",
    "SPREAD_model_score",
    "TOTAL_model_score",
    "MONEYLINE_model_score"
]

for c in cols:
    if c in d.columns:
        print("\n", c)
        print(d[c].head(10).tolist())
    else:
        print("\nMISSING:", c)
