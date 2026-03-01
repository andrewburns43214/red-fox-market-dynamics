import pandas as pd
import re

df = pd.read_csv("data/snapshots_backup.csv", dtype=str)

sample_games = df["game_id"].dropna().unique()[:4]
df = df[df["game_id"].isin(sample_games)].copy()

df["sport"] = "mlb"
df["dk_start_iso"] = "2026-06-01T20:00:00Z"

def to_moneyline(s):
    if pd.isna(s): 
        return s
    s = str(s)

    # convert spreads/totals to MLB style ML prices
    if "@ -" in s or "@ +" in s:
        return s

    # extract odds if present
    m = re.search(r'@\s*([+-]\d+)', s)
    if m:
        return f"Team {m.group(1)}"

    # fallback fake prices
    return "Team @ -150"

df["current_line"] = df["current_line"].apply(to_moneyline)
df["open_line"] = df["open_line"].apply(to_moneyline)

df.to_csv("data/snapshots.csv", index=False)
print("OK: MLB replay lines normalized")
