import pandas as pd

row = pd.read_csv("data/row_state.csv")
snap = pd.read_csv("data/snapshots.csv")

row["last_score_ts"] = pd.to_datetime(row["last_score_ts"], errors="coerce", utc=True)
snap["dk_start_iso"] = pd.to_datetime(snap["dk_start_iso"], errors="coerce", utc=True)

kick = snap.groupby(["sport","game_id"])["dk_start_iso"].first().reset_index()

merged = row.merge(kick, on=["sport","game_id"], how="inner")

# Keep only resolved rows (optional but cleaner)
merged = merged[merged["last_score_ts"].notna() & merged["dk_start_iso"].notna()]

merged["minutes_before_kick"] = (
    (merged["dk_start_iso"] - merged["last_score_ts"])
    .dt.total_seconds() / 60
)

print("Average minutes before kickoff:", merged["minutes_before_kick"].mean())
print("Median minutes before kickoff:", merged["minutes_before_kick"].median())
print("Min minutes before kickoff:", merged["minutes_before_kick"].min())
print("Max minutes before kickoff:", merged["minutes_before_kick"].max())
