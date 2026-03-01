import pandas as pd

df = pd.read_csv("data/snapshots.csv")
print("SNAPSHOT ROWS:", len(df))

print("Unique games:", df["game_id"].nunique())
print("Unique sports:", df["sport"].unique())
