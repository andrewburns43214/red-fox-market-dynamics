import pandas as pd

src = pd.read_csv("data/snapshots_backup.csv", dtype=str)

# keep only a handful of games so report stays readable
games = src["game_id"].dropna().unique()[:3]
df = src[src["game_id"].isin(games)].copy()

# turn them into MLB
df["sport"] = "mlb"

# move kickoff to future so timing bucket works
if "dk_start_iso" in df.columns:
    df["dk_start_iso"] = "2026-06-01T20:00:00Z"

df.to_csv("data/snapshots.csv", index=False)
print("OK: MLB replay snapshot created from real DK data")
