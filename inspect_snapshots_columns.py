import pandas as pd

d = pd.read_csv("data/snapshots.csv")

print("\n=== SNAPSHOTS COLUMN LIST ===")
for c in d.columns:
    print(c)

print("\nTotal Columns:", len(d.columns))
print("Total Rows:", len(d))
