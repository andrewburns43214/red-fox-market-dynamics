import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# import the functions directly from main.py
import main

snaps = pd.read_csv("data/snapshots.csv", keep_default_na=False, dtype=str)
snaps = snaps[snaps["sport"].notna() & (snaps["game"].notna())].copy()

# pick recent sports present
sports = sorted(snaps["sport"].unique().tolist())
print("sports in snapshots:", sports)

# focus on a sport most likely to have finals today
# (you can change this to nba/nhl/ncaab depending on what actually finished)
for sport in ["ncaab","nba","nhl","nfl","ncaaf","mlb"]:
    if sport not in sports:
        continue
    games = sorted(set(snaps[snaps["sport"]==sport]["game"].tolist()))
    games = [g for g in games if g and g.lower() != "nan"]
    print(f"\n=== {sport} games in snapshots: {len(games)} ===")
    sample = games[:10]
    print("sample:", sample)

    finals = main.get_espn_finals_map(sport, games)
    print("finals returned:", len(finals))
    if len(finals):
        # print a few keys so we can see format
        for i,(k,v) in enumerate(list(finals.items())[:10]):
            print("  ", k, "->", v)
