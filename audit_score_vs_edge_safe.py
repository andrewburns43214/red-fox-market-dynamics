import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

print("\n=== LOW EDGE HIGH SCORE CHECK (SAFE) ===\n")

# Safely coerce numeric
d["_net_edge_num"] = pd.to_numeric(d["net_edge"], errors="coerce")

for m in ["SPREAD","MONEYLINE","TOTAL"]:
    sc = f"{m}_model_score"
    if sc in d.columns:
        score_num = pd.to_numeric(d[sc], errors="coerce")
        test = d[(score_num >= 68) & (d["_net_edge_num"] < 5)]
        if len(test) > 0:
            print(f"\nPotential inflation in {m}:")
            print(test[[sc,"net_edge"]].head(10))
        else:
            print(f"\n{m}: no low-edge inflation detected")
