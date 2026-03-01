import pandas as pd

snap = pd.read_csv("data/snapshots.csv", dtype=str)
hist = pd.read_csv("data/final_scores_history.csv", dtype=str)

recent_ids = set(snap.sort_values("timestamp").tail(300)["game_id"])
hist_ids = set(hist["game_id"])

print("Recent game_ids:", len(recent_ids))
print("History game_ids:", len(hist_ids))
print("Overlap:", len(recent_ids & hist_ids))
print("Recent missing finals:", list(sorted(recent_ids - hist_ids))[:15])
