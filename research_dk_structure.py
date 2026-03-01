import pandas as pd

d = pd.read_csv("data/snapshots.csv", dtype=str)

print("COLUMNS:")
print(d.columns.tolist())

print("\nSAMPLE ROWS:")
print(d[["sport","game","dk_start_iso"]].drop_duplicates().head(10).to_string())
