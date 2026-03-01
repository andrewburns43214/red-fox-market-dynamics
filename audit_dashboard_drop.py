import pandas as pd
from datetime import datetime, timezone, timedelta

df = pd.read_csv("data/snapshots.csv", keep_default_na=False)
df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
df["kick"] = pd.to_datetime(df.get("dk_start_iso",""), errors="coerce", utc=True)

print("ROWS BY SPORT (raw snapshots):")
print(df["sport"].value_counts(), "\n")

print("MAX SNAPSHOT TIMESTAMP BY SPORT:")
print(df.groupby("sport")["ts"].max(), "\n")

# 1) what "latest overall timestamp" slice contains
latest_ts = df["ts"].max()
latest_overall = df[df["ts"] == latest_ts].copy()
print("LATEST OVERALL TIMESTAMP:", latest_ts)
print("LATEST OVERALL SPORTS:")
print(latest_overall["sport"].value_counts(), "\n")

# 2) per-sport latest timestamp slice
latest_per_sport = df[df["ts"] == df.groupby("sport")["ts"].transform("max")].copy()
print("LATEST PER-SPORT SLICE SPORTS:")
print(latest_per_sport["sport"].value_counts(), "\n")

# 3) kickoff sanity
print("KICKOFF NaT BY SPORT:")
print(df.groupby("sport")["kick"].apply(lambda x: x.isna().sum()), "\n")

print("KICKOFF MIN/MAX BY SPORT:")
print(df.groupby("sport")["kick"].agg(["min","max"]), "\n")

# 4) show how many are already in the past (common reason to prune)
now = datetime.now(timezone.utc)
df["_past_kick"] = df["kick"].notna() & (df["kick"] < now)
print("PAST KICKOFF ROWS BY SPORT:")
print(df.groupby("sport")["_past_kick"].sum(), "\n")

# 5) show newest 5 rows for NBA, to confirm kick/ts alignment
nba = df[df["sport"]=="nba"].sort_values("ts", ascending=False).head(10)[["timestamp","dk_start_iso","game","side","market"]]
print("NBA newest rows (ts, kick, game/side/market):")
print(nba.to_string(index=False))
