import pandas as pd
from pathlib import Path

GAME_ID = "33701386"

def load_csv(p):
    if not Path(p).exists():
        print("MISSING:", p); return None
    return pd.read_csv(p, dtype=str)

snap = load_csv("data/snapshots.csv")
dash = load_csv("data/dashboard.csv")
state = load_csv("data/row_state.csv")

print("\n=== SNAPSHOTS: game_id presence ===")
if snap is not None and "game_id" in snap.columns:
    s = snap[snap["game_id"] == GAME_ID].copy()
    print("rows:", len(s))
    if len(s):
        cols = [c for c in ["snapshot_id","sport","dk_start_iso","market_display","side","row_status","model_score"] if c in s.columns]
        print(s[cols].head(30).to_string(index=False))
        if "snapshot_id" in s.columns:
            # show which snapshot_ids it appears in
            print("\nunique snapshot_id count:", s["snapshot_id"].nunique())
            print("max snapshot_id:", s["snapshot_id"].max())
else:
    print("snapshots.csv missing game_id column")

print("\n=== DASHBOARD: game_id presence ===")
if dash is not None and "game_id" in dash.columns:
    d = dash[dash["game_id"] == GAME_ID].copy()
    print("rows:", len(d))
    if len(d):
        cols = [c for c in ["sport","game_id","market_display","favored_side","game_confidence","net_edge_market","decision","kickoff","dk_start_iso","timing_bucket"] if c in d.columns]
        print(d[cols].head(30).to_string(index=False))
else:
    print("dashboard.csv missing game_id column")

print("\n=== ROW_STATE: game_id presence ===")
if state is not None and "game_id" in state.columns:
    r = state[state["game_id"] == GAME_ID].copy()
    print("rows:", len(r))
    if len(r):
        cols = [c for c in ["sport","game_id","market_display","side","model_score","last_score","peak_score","timing_bucket","ts"] if c in r.columns]
        print(r[cols].tail(30).to_string(index=False))
else:
    print("row_state.csv missing game_id column")
