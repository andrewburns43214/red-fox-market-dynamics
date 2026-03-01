import pandas as pd
GAME="Utah State @ San Diego State"
snap=pd.read_csv("data/snapshots.csv", dtype=str)
dash=pd.read_csv("data/dashboard.csv", dtype=str)

print("=== Dashboard rows ===")
print(dash[dash["game"]==GAME].to_string(index=False))

print("\n=== Snapshot rows (current run only) ===")
print(snap[snap["game"]==GAME][["timestamp","market","side","bets_pct","money_pct","open_line","current_line"]].to_string(index=False))
