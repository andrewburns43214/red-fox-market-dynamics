import pandas as pd

s = pd.read_csv("data/snapshots.csv")

duke_rows = s[s["side"].str.contains("Duke", case=False, na=False)]

print("Duke rows found:", len(duke_rows))
print(duke_rows[["game_id","sport","game","side","market"]].drop_duplicates())
