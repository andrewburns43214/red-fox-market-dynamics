import pandas as pd

d = pd.read_csv("data/dashboard.csv")

cols = [c for c in d.columns if "odds" in c.lower() or "line" in c.lower()]
print("COLUMNS:")
print(cols)

print("\nSAMPLE:")
print(d[cols].head(10).to_string())
