import pandas as pd

dash = pd.read_csv("data/dashboard.csv")
res  = pd.read_csv("data/results_resolved.csv")

print("[probe] dashboard rows:", len(dash), "cols:", len(dash.columns))
print("[probe] results rows:", len(res), "cols:", len(res.columns))

# normalize
dash["market_display"] = dash["market_display"].astype(str).str.strip().str.upper()
res["market_display"]  = res["market_display"].astype(str).str.strip().str.upper()

dash["favored_side_raw"] = dash["favored_side"].astype(str).str.strip()
res["side_raw"] = res["side"].astype(str).str.strip()

def norm_side(s: str) -> str:
    s = (s or "").strip()
    # common DK-ish pattern: "TEAM @ -270" -> "TEAM"
    if " @ " in s:
        s = s.split(" @ ", 1)[0].strip()
    return s

dash["favored_side_norm"] = dash["favored_side_raw"].map(norm_side)
res["side_norm"] = res["side_raw"].map(norm_side)

# resolved only
res["result"] = res["result"].astype(str).str.strip().str.upper()
resolved = res[res["result"].isin(["WIN","LOSS","PUSH"])].copy()

print("[probe] resolved rows:", len(resolved))
print("[probe] resolved by market:\n", resolved["market_display"].value_counts(dropna=False).to_string())

# join using normalized side
j = dash.merge(
    resolved,
    left_on=["game_id","market_display","favored_side_norm"],
    right_on=["game_id","market_display","side_norm"],
    how="left",
    suffixes=("_dash","_res")
)

matched = j["result"].notna().sum()
print("[probe] join matched:", matched, "/", len(j), f"({matched/len(j):.1%})")

# show top mismatch patterns for moneyline
ml = j[j["market_display"]=="MONEYLINE"].copy()
ml_miss = ml[ml["result"].isna()].copy()
print("[probe] moneyline rows:", len(ml), "unmatched:", len(ml_miss))
if len(ml_miss):
    print("\n[probe] sample unmatched MONEYLINE favored_side vs any candidate sides in results (same game_id):")
    for gid in ml_miss["game_id"].head(10).tolist():
        favs = ml_miss[ml_miss["game_id"]==gid]["favored_side_raw"].unique().tolist()
        cand = res[(res["game_id"]==gid) & (res["market_display"]=="MONEYLINE")]["side_raw"].unique().tolist()
        print(" game_id", gid, "| favored:", favs[:2], "| results sides:", cand[:4])

# sanity: dashboard fields present
need = ["sport","game_id","market_display","favored_side","game_confidence","net_edge","game_decision"]
missing = [c for c in need if c not in dash.columns]
print("[probe] missing expected dashboard fields:", missing)
