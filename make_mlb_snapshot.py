import pandas as pd
from datetime import datetime, timezone

template_cols = [
    "timestamp","sport","game_id","game","side","market",
    "bets_pct","money_pct","open_line","current_line",
    "injury_news","key_number_note","dk_start_iso"
]

def row(ts,sport,gid,game,side,market,bets,money,line):
    return {
        "timestamp": ts,
        "sport": sport,
        "game_id": gid,
        "game": game,
        "side": side,
        "market": market,
        "bets_pct": bets,
        "money_pct": money,
        "open_line": line,
        "current_line": line,
        "injury_news": "",
        "key_number_note": "",
        "dk_start_iso": "2026-02-20T23:00:00Z"
    }

ts = datetime.now(timezone.utc).isoformat()
rows = []

g="Red Sox @ Yankees"
rows += [
row(ts,"mlb","MLB1",g,"Yankees","moneyline","42","68","Yankees @ -160"),
row(ts,"mlb","MLB1",g,"Red Sox","moneyline","58","32","Red Sox @ +140"),
row(ts,"mlb","MLB1",g,"Yankees -1.5","spread","35","61","Yankees -1.5 @ +120"),
row(ts,"mlb","MLB1",g,"Red Sox +1.5","spread","65","39","Red Sox +1.5 @ -140"),
row(ts,"mlb","MLB1",g,"Over 8.5","total","55","70","O 8.5"),
row(ts,"mlb","MLB1",g,"Under 8.5","total","45","30","U 8.5"),
]

df = pd.DataFrame(rows,columns=template_cols)
df.to_csv("data/snapshots.csv",index=False)

print("VALID MLB SNAPSHOT CREATED:",len(df),"rows")
