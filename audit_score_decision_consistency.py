import pandas as pd
d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

print("\n==== SCORE VS DECISION AUDIT ====\n")

for m in ["SPREAD","MONEYLINE","TOTAL"]:
    sc = f"{m}_model_score"
    dc = f"{m}_decision"
    if sc in d.columns and dc in d.columns:
        print(f"\n--- {m} ---")
        print(d[[sc,dc]].head(10))
