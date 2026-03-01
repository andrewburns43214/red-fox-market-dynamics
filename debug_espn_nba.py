from main import get_espn_finals_map
import pandas as pd

df = pd.read_csv("data/snapshots.csv", dtype=str)
nba_games = sorted(set(df[df["sport"]=="nba"]["game"].dropna().astype(str).tolist()))

print("NBA games in snapshot:", len(nba_games))

finals = get_espn_finals_map("nba", nba_games)

print("NBA finals returned:", len(finals))
print("Sample keys:", list(finals.keys())[:10])
