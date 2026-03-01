import pandas as pd
d = pd.read_csv("data/dashboard.csv")

cols = [c for c in d.columns if "odds" in c.lower()]
print("ODDS COLUMNS:", cols)
print("\nSAMPLE ROW:")
print(d.head(1).T)
