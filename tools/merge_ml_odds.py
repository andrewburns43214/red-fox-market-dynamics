import pandas as pd
import re

dash = pd.read_csv("data/dashboard.csv", dtype=str, keep_default_na=False)
snap = pd.read_csv("data/snapshots.csv", dtype=str, keep_default_na=False)

dash["market_display"] = dash["market_display"].astype(str).str.upper()

# latest snapshot per (sport,game_id,market,side) — but market type is NOT in snap["market"]
snap = snap.sort_values("timestamp")
snap = snap.drop_duplicates(subset=["sport","game_id","market","side"], keep="last")

def infer_market_display(side: str, current_line: str) -> str:
    s = (side or "").strip()
    cl = (current_line or "").strip()

    # TOTAL
    if s.startswith("Over ") or s.startswith("Under "):
        return "TOTAL"

    # SPREAD: team +/-number appears in side (most reliable)
    if re.search(r"\s[+-]\d+(\.\d+)?$", s):
        return "SPREAD"

    # Otherwise treat as MONEYLINE
    return "MONEYLINE"

snap["market_display_inferred"] = snap.apply(lambda r: infer_market_display(r.get("side",""), r.get("current_line","")), axis=1)

def parse_price(x: str) -> str:
    m = re.search(r"@\s*([+-]\d+)", str(x))
    return m.group(1) if m else ""

# MONEYLINE odds from inferred ML rows
ml_snap = snap[snap["market_display_inferred"] == "MONEYLINE"].copy()
ml_snap["ml_price"] = ml_snap["current_line"].apply(parse_price)

ml_dash = dash[dash["market_display"] == "MONEYLINE"].copy()

out = ml_dash.merge(
    ml_snap[["sport","game_id","side","ml_price","current_line"]],
    how="left",
    left_on=["sport","game_id","favored_side"],
    right_on=["sport","game_id","side"],
    suffixes=("_dash","_snap"),
)

# Make display sane
out["ml_price"] = out["ml_price"].fillna("")

# current_line might be named "current_line" OR "current_line_snap" depending on collisions
if "current_line_snap" in out.columns:
    out["current_line_snap"] = out["current_line_snap"].fillna("")
elif "current_line" in out.columns:
    out["current_line_snap"] = out["current_line"].fillna("")
else:
    out["current_line_snap"] = ""

cols = ["_game_time","sport","game","favored_side","game_confidence","net_edge","game_decision","ml_price","current_line_snap"]
cols = [c for c in cols if c in out.columns]

print(out[cols].sort_values("game_confidence", ascending=False).to_string(index=False))

matched = int((out["ml_price"].astype(str).str.len() > 0).sum())
print(f"\nML dash rows: {len(ml_dash)} | matched ML odds: {matched}")
print("Snapshot inferred market counts:\n", snap["market_display_inferred"].value_counts().to_string())
