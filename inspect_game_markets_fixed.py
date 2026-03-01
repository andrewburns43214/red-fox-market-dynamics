import pandas as pd

s = pd.read_csv("data/snapshots.csv")

g = "33695178"

print(s[s["game_id"] == g][["market","side"]])
