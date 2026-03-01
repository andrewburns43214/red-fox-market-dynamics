import pandas as pd
d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

print("\n=== STRONG Integrity Check ===\n")

for m in ["SPREAD","MONEYLINE","TOTAL"]:
    sc = f"{m}_model_score"
    dc = f"{m}_decision"
    if sc in d.columns and dc in d.columns:
        strong = d[d[dc] == "STRONG"]
        if len(strong) > 0:
            print(f"\n{m} STRONG rows:")
            print(strong[[sc,"net_edge"]].head(10))
        else:
            print(f"\n{m}: no STRONG rows")
