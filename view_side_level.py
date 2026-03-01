import pandas as pd

s = pd.read_csv("data/snapshots.csv")

# take only latest snapshot per (sport, game_id, market_display, side)
# assuming snapshots has snapshot_id or snapshot timestamp
s = s.sort_values("logged_at_utc")
latest = s.groupby(["sport","game_id","market_display","side"]).tail(1)

print("\nSIDE-LEVEL VIEW SAMPLE:\n")
print(latest[["sport","game_id","market_display","side"]].head(20))
print("\nTOTAL SIDE-LEVEL ROWS:", len(latest))
