import pandas as pd

sn = pd.read_csv("data/snapshots.csv", dtype=str)
hi = pd.read_csv("data/final_scores_history.csv", dtype=str)

# attach sport + any game label if present
sn_g = sn[["game_id","sport"]].drop_duplicates()
j = hi.merge(sn_g, on="game_id", how="left")

print("[history] rows:", len(hi))
print("[history] sports distribution (inferred from snapshots):")
print(j["sport"].value_counts(dropna=False).to_string())

print("\n[history] games:")
cols = [c for c in ["sport","game_id","team1","team1_score","team2","team2_score","resolved_at_utc"] if c in j.columns]
print(j[cols].to_string(index=False))
