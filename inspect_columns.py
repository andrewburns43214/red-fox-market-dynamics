import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("\nALL COLUMNS:")
for c in d.columns:
    print(c)
