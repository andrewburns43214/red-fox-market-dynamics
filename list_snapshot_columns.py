import pandas as pd

s = pd.read_csv("data/snapshots.csv", nrows=5)

print("\n=== SNAPSHOTS COLUMNS ===\n")
for c in s.columns:
    print(c)

print("\nTotal columns:", len(s.columns))
