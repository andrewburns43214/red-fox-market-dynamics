import pandas as pd

d = pd.read_csv("data/finals_espn.csv", nrows=5)

print("COLUMNS:")
for c in d.columns:
    print("-", c)

print("\nSAMPLE:")
print(d.head(3).to_string())
