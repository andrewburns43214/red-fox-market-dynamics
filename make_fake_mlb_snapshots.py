import pandas as pd
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc).replace(microsecond=0)
kick = now + timedelta(hours=5)

rows = [
    # GAME 1: Yankees @ Red Sox
    dict(sport="mlb", game_id="MLB_TEST1", game="Yankees @ Red Sox", side="Yankees", market="moneyline",
         bets_pct=40, money_pct=65, current_line="Yankees @ -160", open_line="Yankees @ -155"),
    dict(sport="mlb", game_id="MLB_TEST1", game="Yankees @ Red Sox", side="Red Sox", market="moneyline",
         bets_pct=60, money_pct=35, current_line="Red Sox @ +140", open_line="Red Sox @ +135"),
    dict(sport="mlb", game_id="MLB_TEST1", game="Yankees @ Red Sox", side="Red Sox +1.5", market="spread",
         bets_pct=72, money_pct=52, current_line="Red Sox +1.5 @ -150", open_line="Red Sox +1.5 @ -145"),

    # GAME 2: Dodgers @ Giants
    dict(sport="mlb", game_id="MLB_TEST2", game="Dodgers @ Giants", side="Dodgers", market="moneyline",
         bets_pct=45, money_pct=70, current_line="Dodgers @ -170", open_line="Dodgers @ -150"),
    dict(sport="mlb", game_id="MLB_TEST2", game="Dodgers @ Giants", side="Giants +1.5", market="spread",
         bets_pct=80, money_pct=60, current_line="Giants +1.5 @ -170", open_line="Giants +1.5 @ -140"),
]

out = []
for r in rows:
    out.append({
        # engine expects these
        "timestamp": now.isoformat().replace("+00:00","Z"),
        "sport": r["sport"],
        "game_id": r["game_id"],
        "game": r["game"],
        "side": r["side"],
        "market": r["market"],
        "current_line": r["current_line"],
        "open_line": r["open_line"],
        "bets_pct": r["bets_pct"],
        "money_pct": r["money_pct"],
        "dk_start_iso": kick.isoformat().replace("+00:00","Z"),

        # compatibility aliases (harmless if unused)
        "selection": r["side"],
        "bet_pct": r["bets_pct"],
        "handle_pct": r["money_pct"],
        "current": r["current_line"],
        "open": r["open_line"],
    })

pd.DataFrame(out).to_csv("data/snapshots.csv", index=False)
print("OK: wrote data/snapshots.csv in engine format (fake MLB)")
