import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

now = datetime.now(timezone.utc)

def load(p):
    if not Path(p).exists():
        print("MISSING:", p); return None
    return pd.read_csv(p, dtype=str)

snap = load("data/snapshots.csv")
dash = load("data/dashboard.csv")

if snap is None or dash is None:
    raise SystemExit()

# pick a kickoff-like column from snapshots
kick_col = None
for c in ["dk_start_iso","start_iso","kickoff_iso","game_time_iso"]:
    if c in snap.columns:
        kick_col = c; break

print("snap kickoff col:", kick_col)
if kick_col is None:
    print("No kickoff column in snapshots to audit time-window.")
    raise SystemExit()

# parse UTC timestamps
s = snap.copy()
s["_dt"] = pd.to_datetime(s[kick_col], errors="coerce", utc=True)
future = s[s["_dt"].notna() & (s["_dt"] > now)].copy()

print("future snapshot rows:", len(future))

# ensure join keys exist
need = ["sport","game_id"]
for k in need:
    if k not in future.columns or k not in dash.columns:
        print("Missing join key:", k)
        raise SystemExit()

# unique games in future snapshots
future_games = future[["sport","game_id"]].drop_duplicates()

dash_games = dash[["sport","game_id"]].drop_duplicates()

merged = future_games.merge(dash_games, on=["sport","game_id"], how="left", indicator=True)
missing = merged[merged["_merge"]=="left_only"].copy()

print("future games missing from dashboard:", len(missing))
if len(missing):
    print(missing.head(50).to_string(index=False))
