import pandas as pd

s = pd.read_csv("data/snapshots.csv")

print(s["game_id"].dtype)
print(type(s["game_id"].iloc[0]))
