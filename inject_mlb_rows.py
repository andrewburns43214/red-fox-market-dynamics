import pandas as pd
from datetime import datetime, timezone, timedelta

path = "data/snapshots.csv"
df = pd.read_csv(path, dtype=str)

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
future = (datetime.now(timezone.utc)+timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")

rows = [
# -------- Price Band Test (-160 stable, dog popular RL) --------
dict(timestamp=now,sport="mlb",game_id="MLB_TEST1",game="Yankees @ Red Sox",side="Yankees",market="MONEYLINE",bets_pct="40",money_pct="65",open_line="Yankees @ -160",current_line="Yankees @ -160",injury_news="",key_number_note="",dk_start_iso=future),
dict(timestamp=now,sport="mlb",game_id="MLB_TEST1",game="Yankees @ Red Sox",side="Red Sox",market="MONEYLINE",bets_pct="60",money_pct="35",open_line="Red Sox @ +140",current_line="Red Sox @ +140",injury_news="",key_number_note="",dk_start_iso=future),
dict(timestamp=now,sport="mlb",game_id="MLB_TEST1",game="Yankees @ Red Sox",side="Red Sox +1.5",market="SPREAD",bets_pct="72",money_pct="52",open_line="Red Sox +1.5 @ -150",current_line="Red Sox +1.5 @ -150",injury_news="",key_number_note="",dk_start_iso=future),

# -------- True Contradiction Test --------
dict(timestamp=now,sport="mlb",game_id="MLB_TEST2",game="Dodgers @ Giants",side="Dodgers",market="MONEYLINE",bets_pct="45",money_pct="70",open_line="Dodgers @ -150",current_line="Dodgers @ -170",injury_news="",key_number_note="",dk_start_iso=future),
dict(timestamp=now,sport="mlb",game_id="MLB_TEST2",game="Dodgers @ Giants",side="Giants +1.5",market="SPREAD",bets_pct="80",money_pct="60",open_line="Giants +1.5 @ -140",current_line="Giants +1.5 @ -170",injury_news="",key_number_note="",dk_start_iso=future),

# -------- ML Anchor Test --------
dict(timestamp=now,sport="mlb",game_id="MLB_TEST3",game="Mets @ Marlins",side="Mets",market="MONEYLINE",bets_pct="48",money_pct="72",open_line="Mets @ -145",current_line="Mets @ -145",injury_news="",key_number_note="",dk_start_iso=future),
dict(timestamp=now,sport="mlb",game_id="MLB_TEST3",game="Mets @ Marlins",side="Marlins +1.5",market="SPREAD",bets_pct="75",money_pct="55",open_line="Marlins +1.5 @ -155",current_line="Marlins +1.5 @ -170",injury_news="",key_number_note="",dk_start_iso=future),

# -------- Late RL Public Tax --------
dict(timestamp=now,sport="mlb",game_id="MLB_TEST4",game="Braves @ Phillies",side="Braves",market="MONEYLINE",bets_pct="50",money_pct="68",open_line="Braves @ -170",current_line="Braves @ -170",injury_news="",key_number_note="",dk_start_iso=future),
dict(timestamp=now,sport="mlb",game_id="MLB_TEST4",game="Braves @ Phillies",side="Phillies +1.5",market="SPREAD",bets_pct="85",money_pct="60",open_line="Phillies +1.5 @ -165",current_line="Phillies +1.5 @ -200",injury_news="",key_number_note="",dk_start_iso=future),
]

df2 = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
df2.to_csv(path, index=False)

print("Injected synthetic MLB rows:", len(rows))
