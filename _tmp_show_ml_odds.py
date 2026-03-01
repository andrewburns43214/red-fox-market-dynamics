import pandas as pd, re

dash = pd.read_csv("data/dashboard.csv", dtype=str, keep_default_na=False)
snap = pd.read_csv("data/snapshots.csv", dtype=str, keep_default_na=False)

dash["market_display"] = dash["market_display"].astype(str).str.upper()
snap["market"] = snap["market"].astype(str).str.upper()

# Use latest snapshot row per (sport, game_id, market, side)
snap = snap.sort_values("timestamp")
snap = snap.drop_duplicates(subset=["sport","game_id","market","side"], keep="last")

def infer_market(side: str) -> str:
    s = (side or "").strip()
    if s.startswith("Over ") or s.startswith("Under "):
        return "TOTAL"
    if re.search(r"\s[+-]\d+(\.\d+)?$", s):
        return "SPREAD"
    return "MONEYLINE"

snap["market_display_inferred"] = snap["side"].apply(infer_market)

def parse_price(x) -> str:
    m = re.search(r"@\s*([+-]\d+)", str(x))
    return m.group(1) if m else ""

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

# Normalize display columns deterministically (no .get() string defaults)
out["ml_price"] = out["ml_price"].fillna("")
if "current_line_snap" in out.columns:
    out["ml_current_line"] = out["current_line_snap"].fillna("")
elif "current_line" in out.columns:
    out["ml_current_line"] = out["current_line"].fillna("")
else:
    out["ml_current_line"] = ""

cols = ["_game_time","sport","game","favored_side","game_confidence","net_edge","game_decision","ml_price","ml_current_line"]
cols = [c for c in cols if c in out.columns]

print(out[cols].sort_values("game_confidence", ascending=False).to_string(index=False))

matched = int((out["ml_price"].astype(str).str.len() > 0).sum())
print(f"\nML dash rows: {len(ml_dash)} | matched ML odds: {matched}")
print("\nSnapshot inferred market counts:\n" + snap["market_display_inferred"].value_counts().to_string())
