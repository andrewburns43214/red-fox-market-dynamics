import pandas as pd

sn = pd.read_csv("data/snapshots.csv", dtype=str)

completed = sn[
    sn["final_score_for"].notna() &
    sn["final_score_for"].astype(str).str.strip().ne("")
]

g = (
    completed
    .groupby(["sport","game_id"])
    .size()
    .reset_index(name="row_count")
)

print(g.sort_values("row_count").head(10))
print("\nGames with only 1 row:")
print(g[g["row_count"] == 1].head(10))
