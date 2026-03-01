import pandas as pd
d = pd.read_csv("data/snapshots.csv", dtype=str)
print("snap rows:", len(d))
print("has dk_start_iso:", "dk_start_iso" in d.columns)

g = d[["sport","game","dk_start_iso"]].drop_duplicates()
print("\nUNIQUE games:", len(g))
print(g.head(15).to_string(index=False))
