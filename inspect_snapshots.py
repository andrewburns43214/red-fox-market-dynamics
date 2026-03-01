import pandas as pd

snap = pd.read_csv("data/snapshots.csv")

print("Total snapshot rows:", len(snap))
print("\nSports in snapshots:")
print(snap["sport"].value_counts())

print("\nRecent NCAAB snapshot rows:")
n = snap[snap["sport"]=="ncaab"]
print("NCAAB rows:", len(n))
print(n.tail(10)[["sport","game_id","market_display","side"]])
