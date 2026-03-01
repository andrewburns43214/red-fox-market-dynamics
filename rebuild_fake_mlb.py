import pandas as pd
from datetime import datetime, timedelta, timezone

p = "data/snapshots.csv"
df = pd.read_csv(p, keep_default_na=False, dtype=str)

# Remove ALL existing MLB rows
df = df[df["sport"].str.upper()!="MLB"]

now = datetime.now(timezone.utc)
start = (now - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")

games = [
    ("Yankees","Red Sox",-160,140),
    ("Dodgers","Giants",-170,150),
    ("Mets","Braves",-135,125),
]

rows=[]

for a,b,fa,da in games:

    gid=f"{a}_{b}"

    # MONEYLINE
    rows += [
        dict(sport="MLB",game=f"{a} @ {b}",game_id=gid,market="moneyline",
             side=a,side_key=a,dk_start_iso=start,current_line=f"{a} @ {fa:+d}",
             current_line_val="",current_odds=str(fa),money_pct="55",bets_pct="45",timestamp=start),

        dict(sport="MLB",game=f"{a} @ {b}",game_id=gid,market="moneyline",
             side=b,side_key=b,dk_start_iso=start,current_line=f"{b} @ {da:+d}",
             current_line_val="",current_odds=str(da),money_pct="45",bets_pct="55",timestamp=start),
    ]

    # RUN LINE (spread)
    rows += [
        dict(sport="MLB",game=f"{a} @ {b}",game_id=gid,market="spread",
             side=a,side_key=f"{a}_RL",dk_start_iso=start,current_line=f"{a} -1.5 @ -110",
             current_line_val="-1.5 @ -110",current_odds="-110",money_pct="40",bets_pct="60",timestamp=start),

        dict(sport="MLB",game=f"{a} @ {b}",game_id=gid,market="spread",
             side=b,side_key=f"{b}_RL",dk_start_iso=start,current_line=f"{b} +1.5 @ -110",
             current_line_val="+1.5 @ -110",current_odds="-110",money_pct="60",bets_pct="40",timestamp=start),
    ]

df = pd.concat([df,pd.DataFrame(rows)],ignore_index=True)
df.to_csv(p,index=False)

print("Rebuilt MLB mock rows:",len(rows))
