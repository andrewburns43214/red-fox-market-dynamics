import pandas as pd

s = pd.read_csv("data/snapshots.csv")

print("SNAPSHOT COLUMNS:")
for c in s.columns:
    print(c)
