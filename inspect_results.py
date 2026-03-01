import pandas as pd

r = pd.read_csv("data/results_resolved.csv")

print("COLUMNS:")
for c in r.columns:
    print("-", c)

print("\nHead:")
print(r.head(5))
