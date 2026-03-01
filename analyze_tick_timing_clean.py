import pandas as pd

row = pd.read_csv("data/row_state.csv")
snap = pd.read_csv("data/snapshots.csv")
res = pd.read_csv("data/results_resolved.csv")

row["last_score_ts"] = pd.to_datetime(row["last_score_ts"], errors="coerce", utc=True)
snap["dk_start_iso"] = pd.to_datetime(snap["dk_start_iso"], errors="coerce", utc=True)

# kickoff per game
kick = snap.groupby(["sport","game_id"])["dk_start_iso"].first().reset_index()

# anchor to resolved rows (canonical)
base = res.merge(
    row[["sport","game_id","side","last_score_ts"]],
    on=["sport","game_id","side"],
    how="left"
)

base = base.merge(
    kick,
    on=["sport","game_id"],
    how="left"
)

base = base[
    base["last_score_ts"].notna() &
    base["dk_start_iso"].notna()
]

base["minutes_before_kick"] = (
    (base["dk_start_iso"] - base["last_score_ts"])
    .dt.total_seconds() / 60
)

print("Row count used:", len(base))
print("Average minutes before kickoff:", base["minutes_before_kick"].mean())
print("Median minutes before kickoff:", base["minutes_before_kick"].median())
print("Min minutes before kickoff:", base["minutes_before_kick"].min())
print("Max minutes before kickoff:", base["minutes_before_kick"].max())
