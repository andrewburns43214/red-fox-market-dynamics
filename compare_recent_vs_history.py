import pandas as pd

snap = pd.read_csv("data/snapshots.csv", dtype=str)
hist = pd.read_csv("data/final_scores_history.csv", dtype=str)

latest_ids = set(snap.sort_values("timestamp").tail(200)["game_id"])
hist_ids = set(hist["game_id"])

print("Recent snapshot games:")
print(sorted(list(latest_ids))[:10])

print("\nHistory games:")
print(sorted(list(hist_ids)))

print("\nRecent games missing finals:")
print(latest_ids - hist_ids)
