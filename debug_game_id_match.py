import pandas as pd

s = pd.read_csv("data/snapshots.csv", dtype=str)
b = pd.read_csv("data/signals_baseline.csv", dtype=str)

print("SNAP game_id sample:", s["game_id"].dropna().unique()[:5])
print("BASE game_id sample:", b["game_id"].dropna().unique()[:5])

snap_ids = set(s["game_id"].dropna())
base_ids = set(b["game_id"].dropna())

print("Baseline IDs not in snapshots:", len(base_ids - snap_ids))
print("Example missing:", list(base_ids - snap_ids)[:5])
