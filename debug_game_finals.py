import pandas as pd

s = pd.read_csv("data/snapshots.csv", dtype=str)

g = s[s["game_id"]=="33701386"]

print("\nRows with finals populated:")
print(g[g["final_score_for"].str.strip()!=""][["side","final_score_for","final_score_against"]])

print("\nRows without finals:")
print(g[g["final_score_for"].str.strip()==""][["side"]])
