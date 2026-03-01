import pandas as pd

s = pd.read_csv("data/snapshots.csv", dtype=str)

mask = (
    (s["sport"].str.lower() == "ncaab") &
    (s["game"] == "Mississippi State @ Alabama")
)

print(s.loc[mask, ["game_id"]].drop_duplicates())
print("Total rows:", mask.sum())
