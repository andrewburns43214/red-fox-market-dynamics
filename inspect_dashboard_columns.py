import pandas as pd

d = pd.read_csv("data/dashboard.csv", dtype=str)

print("COLUMNS:")
for c in d.columns:
    print(c)

print("\nROW COUNT:", len(d))
