import pandas as pd
d = pd.read_csv("data/dashboard.csv")
for m in ["SPREAD","MONEYLINE","TOTAL"]:
    c = f"{m}_decision"
    if c in d.columns:
        print(m, d[c].value_counts(dropna=False).to_dict())
