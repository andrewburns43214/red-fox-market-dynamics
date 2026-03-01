import pandas as pd
sn = pd.read_csv("data/snapshots.csv", dtype=str)
print(sn["sport"].value_counts())
