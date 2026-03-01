import pandas as pd

s = pd.read_csv("data/snapshots.csv", keep_default_na=False)
print("\n[SNAPSHOTS SHAPE] rows:", len(s), "cols:", len(s.columns))

need = ["sport","game_id","game","side","market","bets_pct","money_pct","current_line","dk_start_iso"]
missing = [c for c in need if c not in s.columns]
print("missing:", missing)

# look at latest timestamp slice
if "timestamp" in s.columns:
    ts = s["timestamp"].astype(str)
    latest_ts = ts.max()
    sl = s[ts == latest_ts].copy()
    print("\n[LATEST SNAPSHOT] timestamp:", latest_ts, "rows:", len(sl))
    print("sports:", sl["sport"].value_counts().to_dict() if "sport" in sl.columns else {})
    print("markets:", sl["market"].value_counts().to_dict() if "market" in sl.columns else {})
else:
    print("no timestamp column in snapshots.csv (ok if your feed is different)")

