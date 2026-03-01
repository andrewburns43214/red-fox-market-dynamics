import pandas as pd

d = pd.read_csv("data/decision_snapshots.csv")
r = pd.read_csv("data/results_resolved.csv")

graded = r[r["result"].isin(["WIN","LOSS"])]

print("Decision snapshot rows:", len(d))
print("Graded rows:", len(graded))

test = graded.merge(
    d,
    on=["sport","game_id","market_display","side"],
    how="left",
    suffixes=("","_ds")
)

print("Matched rows:", test["game_decision_ds"].notna().sum())
