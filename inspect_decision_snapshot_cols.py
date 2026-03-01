import pandas as pd
d = pd.read_csv("data/decision_snapshots.csv")
print(d.columns.tolist())
print("Rows:", len(d))
