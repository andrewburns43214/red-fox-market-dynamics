import pandas as pd

d = pd.read_csv("data/snapshots.csv")

print("\nLatest timestamps by sport:")
print(d.groupby("sport")["timestamp"].max())
