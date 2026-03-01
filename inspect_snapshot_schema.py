import pandas as pd
d = pd.read_csv("data/snapshots.csv", nrows=5)
print(d.columns.tolist())
