import pandas as pd

sn = pd.read_csv("data/snapshots.csv", dtype=str)
hi = pd.read_csv("data/final_scores_history.csv", dtype=str)

sn_g = sn[["sport","game_id"]].drop_duplicates()
hi_g = hi[["game_id"]].drop_duplicates()

missing = sn_g[~sn_g["game_id"].isin(set(hi_g["game_id"]))]

print("[history] games in history:", len(hi_g))
print("[snap] games in snapshots:", len(sn_g))
print("[delta] snapshot games missing finals:", len(missing))

print("\n[delta] sample missing game_ids:")
print(missing.head(25).to_string(index=False))
