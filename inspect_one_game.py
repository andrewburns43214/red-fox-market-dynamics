import pandas as pd

game_id = "33706065"

snap = pd.read_csv("data/snapshots.csv", dtype=str)

rows = snap[snap["game_id"] == game_id][["sport","game","side","dk_start_iso"]]
print(rows.head(20))
