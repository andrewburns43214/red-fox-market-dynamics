import pandas as pd
sn = pd.read_csv("data/snapshots.csv", dtype=str)
sn["final_score_for"] = sn.get("final_score_for","").fillna("").astype(str).str.strip()
sn["final_score_against"] = sn.get("final_score_against","").fillna("").astype(str).str.strip()

done = sn[(sn["final_score_for"]!="") & (sn["final_score_against"]!="")].copy()
g = done.groupby("sport")["game_id"].nunique().reset_index(name="games_with_finals").sort_values("games_with_finals", ascending=False)
print(g.to_string(index=False))

print("\nSample game_ids per sport:")
for sp in g["sport"].tolist():
    samp = done[done["sport"]==sp][["sport","game_id","game"]].drop_duplicates().head(5)
    print("\n", samp.to_string(index=False))
