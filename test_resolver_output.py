from main import get_espn_finals_map
import pandas as pd

snap = pd.read_csv("data/snapshots.csv", dtype=str)

sport = "nba"
games = snap[snap["sport"]==sport]["game"].unique().tolist()

finals = get_espn_finals_map(sport, games)

print("Returned finals count:", len(finals))
print("Sample:", list(finals.items())[:5])
