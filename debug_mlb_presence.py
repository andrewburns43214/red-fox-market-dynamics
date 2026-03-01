import pandas as pd
df = pd.read_csv("data/snapshots.csv", dtype=str)

print("TOTAL SNAPSHOT ROWS:", len(df))
print("MLB SNAPSHOT ROWS:", (df["sport"].str.lower()=="mlb").sum())

# show MLB rows
print("\nMLB rows in snapshots:")
print(df[df["sport"].str.lower()=="mlb"][["game_id","market","side","current_line"]])
