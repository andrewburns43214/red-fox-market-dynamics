import pandas as pd

s = pd.read_csv("data/snapshots.csv", dtype=str)

# timestamp parse
s["timestamp_dt"] = pd.to_datetime(s["timestamp"], errors="coerce", utc=True)

# last ~36 hours
cut = pd.Timestamp.utcnow() - pd.Timedelta(hours=36)
s2 = s[s["timestamp_dt"] >= cut].copy()

print("[snap] rows last36h:", len(s2))
print("[snap] unique game_id last36h:", s2["game_id"].nunique())

# show latest timestamp per game
g = (s2.groupby(["sport","game_id"])["timestamp_dt"].max()
       .reset_index()
       .sort_values("timestamp_dt", ascending=False)
       .head(25))
print("\n[snap] latest games seen (top 25):")
print(g.to_string(index=False))
