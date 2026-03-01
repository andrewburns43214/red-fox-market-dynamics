import pandas as pd

snap_path = "data/snapshots.csv"
hist_path = "data/final_scores_history.csv"

snaps = pd.read_csv(snap_path, dtype=str)
history = pd.read_csv(hist_path, dtype=str)

history["team1_score"] = pd.to_numeric(history["team1_score"], errors="coerce")
history["team2_score"] = pd.to_numeric(history["team2_score"], errors="coerce")

merged = snaps.merge(history, on="game_id", how="left")

print("Merged rows:", len(merged))
print("Non-null team1_score in merged:", merged["team1_score"].notna().sum())

scored = merged[merged["team1_score"].notna()]
print("\nSample merged rows with scores:")
print(scored[["sport","game_id","side","team1","team1_score","team2","team2_score"]].head(10))
