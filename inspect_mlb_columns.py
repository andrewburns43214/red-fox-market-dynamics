import pandas as pd
d = pd.read_csv("data/dashboard.csv", dtype=str)
print(sorted(d.columns))
print("\nSAMPLE MLB ROWS:")
print(d[d["sport"].str.lower()=="mlb"].head(5).T)
