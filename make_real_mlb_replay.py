import pandas as pd

df = pd.read_csv("data/snapshots_backup.csv", dtype=str)

# keep a few games only
sample_games = df["game_id"].dropna().unique()[:4]
df = df[df["game_id"].isin(sample_games)].copy()

# convert sport to MLB
df["sport"] = "mlb"

# push kickoff into future so timing bucket works
if "dk_start_iso" in df.columns:
    df["dk_start_iso"] = "2026-06-01T20:00:00Z"

df.to_csv("data/snapshots.csv", index=False)
print("OK: real DK snapshot converted to MLB replay")
