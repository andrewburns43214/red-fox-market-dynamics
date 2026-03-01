import pandas as pd
s = pd.read_csv("data/snapshots.csv")

print("Games in snapshot:", s["game"].nunique())
print("Rows:", len(s))
print("\nMarkets per game:\n")

print(s.groupby("game")["market"].unique())
