import pandas as pd
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc).replace(microsecond=0)
kick = now + timedelta(hours=5)

def fmt_odds(x):
    x = int(x)
    return f"+{x}" if x > 0 else str(x)

rows = [
    ("G1","Yankees","Red Sox",-160,+140,40,65,60,35),
    ("G2","Dodgers","Giants",-170,+150,45,70,55,30),
    ("G3","Mets","Braves",-135,+125,52,55,48,45),
]

out = []
for gid,a,b,a_odds,b_odds,a_bets,a_money,b_bets,b_money in rows:
    game = f"{a} @ {b}"

    out.append(dict(
        timestamp=now.isoformat().replace("+00:00","Z"),
        sport="mlb", game_id=gid, game=game, side=a, market="moneyline",
        current_line=f"{a} @ {fmt_odds(a_odds)}",
        open_line=f"{a} @ {fmt_odds(a_odds+5)}",
        bets_pct=a_bets, money_pct=a_money,
        dk_start_iso=kick.isoformat().replace("+00:00","Z")
    ))

    out.append(dict(
        timestamp=now.isoformat().replace("+00:00","Z"),
        sport="mlb", game_id=gid, game=game, side=b, market="moneyline",
        current_line=f"{b} @ {fmt_odds(b_odds)}",
        open_line=f"{b} @ {fmt_odds(b_odds-5)}",
        bets_pct=b_bets, money_pct=b_money,
        dk_start_iso=kick.isoformat().replace("+00:00","Z")
    ))

pd.DataFrame(out).to_csv("data/snapshots.csv", index=False)
print("OK: TRUE MLB test slate written")
