import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("COLUMNS:")
for c in d.columns:
    print(" -", c)

print("\nSample net_edge values:")
print(d[["game","net_edge"]].head(10))

print("\nUnique net_edge sample (first 20):")
vals = sorted([v for v in d["net_edge"].dropna().unique()])
print(vals[:20])

print("\nPer-market net_edge columns:")
print([c for c in d.columns if "net_edge" in c])
