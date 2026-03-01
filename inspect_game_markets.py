import pandas as pd

s = pd.read_csv("data/snapshots.csv")

g = 33695178  # Duke @ Notre Dame

print("\nMarkets for game_id:", g)
print(s[s["game_id"] == g][["market", "side"]])
