from main import _espn_finals_map_date_range, ESPN_SCOREBOARD_BASE
import pandas as pd

df = pd.read_csv("data/snapshots.csv", dtype=str)
nba_games = sorted(set(df[df["sport"]=="nba"]["game"].dropna().astype(str).tolist()))
nhl_games = sorted(set(df[df["sport"]=="nhl"]["game"].dropna().astype(str).tolist()))
cbb_games = sorted(set(df[df["sport"]=="ncaab"]["game"].dropna().astype(str).tolist()))

print("NBA games:", len(nba_games))
print("NHL games:", len(nhl_games))
print("NCAAB games:", len(cbb_games))

# IMPORTANT: increase days window for finals (ESPN can lag or DK includes earlier games)
# We'll test days=10, and we will not rely on want_keys filtering to confirm ESPN is actually returning finals.
