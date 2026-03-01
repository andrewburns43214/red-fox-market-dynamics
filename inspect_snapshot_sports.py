import pandas as pd

s = pd.read_csv("data/snapshots.csv")

print("Unique sports in snapshots:")
print(s["sport"].unique())

print("\nCount by sport:")
print(s["sport"].value_counts())
