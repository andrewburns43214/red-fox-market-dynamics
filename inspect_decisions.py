import pandas as pd
d = pd.read_csv("data/dashboard.csv", keep_default_na=False)
print("COLUMNS:")
for c in d.columns:
    print("-", c)
print("\nDecision samples:")
for m in ["SPREAD","MONEYLINE","TOTAL"]:
    col = f"{m}_decision"
    if col in d.columns:
        print(col, "=>", list(d[col].head(5)))
    else:
        print(col, "MISSING")
