import pandas as pd

d = pd.read_csv("data/snapshots.csv")

print("\nColumns in snapshots:")
for c in d.columns:
    if "edge" in c.lower():
        print(c)
