import pandas as pd
from datetime import datetime, timezone

rows = [
    # --- Favorite band test (-160 stable, RL popular dog) ---
    dict(sport="mlb", game_id="TEST1", market="MONEYLINE", side="Yankees", current_line="Yankees @ -160", bets_pct="40", money_pct="65"),
    dict(sport="mlb", game_id="TEST1", market="MONEYLINE", side="Red Sox", current_line="Red Sox @ +140", bets_pct="60", money_pct="35"),

    # --- RL contradiction test ---
    dict(sport="mlb", game_id="TEST2", market="MONEYLINE", side="Dodgers", current_line="Dodgers @ -150", bets_pct="45", money_pct="70"),
    dict(sport="mlb", game_id="TEST2", market="RUNLINE", side="Giants +1.5", current_line="Giants +1.5 @ -130", bets_pct="75", money_pct="55"),

    # --- ML anchor test ---
    dict(sport="mlb", game_id="TEST3", market="MONEYLINE", side="Mets", current_line="Mets @ -145", bets_pct="48", money_pct="72"),
    dict(sport="mlb", game_id="TEST3", market="RUNLINE", side="Marlins +1.5", current_line="Marlins +1.5 @ -150", bets_pct="70", money_pct="50"),

    # --- Late RL public tax test ---
    dict(sport="mlb", game_id="TEST4", market="MONEYLINE", side="Braves", current_line="Braves @ -170", bets_pct="50", money_pct="68"),
    dict(sport="mlb", game_id="TEST4", market="RUNLINE", side="Phillies +1.5", current_line="Phillies +1.5 @ -180", bets_pct="82", money_pct="60"),
]

ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

for r in rows:
    r["timestamp"] = ts

df = pd.DataFrame(rows)
df.to_csv("data/snapshots_mlb_test.csv", index=False)
print("created synthetic mlb snapshot file")
