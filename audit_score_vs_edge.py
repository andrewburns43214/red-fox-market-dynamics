import pandas as pd
d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

print("\n=== LOW EDGE HIGH SCORE CHECK ===\n")

for m in ["SPREAD","MONEYLINE","TOTAL"]:
    sc = f"{m}_model_score"
    if sc in d.columns:
        test = d[(d[sc].astype(float) >= 68) & (d["net_edge"].astype(float) < 5)]
        if len(test) > 0:
            print(f"\nPotential inflation in {m}:")
            print(test[[sc,"net_edge"]].head(10))
