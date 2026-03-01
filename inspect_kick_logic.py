import pandas as pd

d = pd.read_csv("data/dashboard.csv")
s = pd.read_csv("data/snapshots.csv")

# simulate lifecycle section
latest = s.groupby(["sport","game_id","market","side"], as_index=False).tail(1).copy()

now_ny = pd.Timestamp.now(tz="America/New_York")

print("NOW_NY:", now_ny)

_kick = pd.to_datetime(latest["dk_start_iso"], errors="coerce", utc=True)

try:
    _kick = _kick.dt.tz_convert("America/New_York")
except Exception:
    pass

latest["_kick"] = _kick

print("\nSample upcoming games:")
print(
    latest.sort_values("_kick")
    [["sport","game_id","game","_kick"]]
    .drop_duplicates("game_id")
    .head(15)
)

print("\nEarliest kickoff:", latest["_kick"].min())
print("Latest kickoff:", latest["_kick"].max())

print("\nCount >= now:", (latest["_kick"] >= now_ny).sum())
print("Total rows:", len(latest))
