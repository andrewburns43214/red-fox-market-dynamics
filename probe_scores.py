import pandas as pd
d = pd.read_csv("data/dashboard.csv", dtype=str)

score_cols = [c for c in d.columns if "model_score" in c.lower()]
print("Score columns:", score_cols)

for c in score_cols:
    print(f"\nColumn: {c}")
    print(d[c].astype(str).value_counts().head(10))
