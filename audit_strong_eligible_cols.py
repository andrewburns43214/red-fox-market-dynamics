import pandas as pd
d = pd.read_csv("data/dashboard.csv")
cols = [c for c in d.columns if c.endswith("_strong_eligible")]
print("elig cols:", cols)
if cols:
    for c in cols:
        print(c, d[c].astype(str).value_counts().head(10).to_dict())
