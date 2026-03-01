import pandas as pd

d = pd.read_csv("data/dashboard.csv")
s = pd.read_csv("data/snapshots.csv")

# normalize ids as strings (your snapshots game_id is object/str)
d["game_id"] = d["game_id"].astype(str)
s["game_id"] = s["game_id"].astype(str)

# games with blank ML in dashboard (NCAAB only)
missing = d[(d["sport"]=="ncaab") & (d["MONEYLINE_model_score"].isna())]
print("NCAAB games with missing MONEYLINE_model_score in dashboard:", len(missing))
print(missing[["game_id","game","dk_start_iso","SPREAD_favored","TOTAL_favored"]].head(25).to_string(index=False))

# helper: detect "team-only" rows (possible ML) in snapshots
def is_possible_ml_side(x: str) -> bool:
    x = str(x or "")
    if x.strip()=="":
        return False
    if "over" in x.lower() or "under" in x.lower():
        return False
    # spread indicators
    if " +" in x or " -" in x:
        return False
    # if it's just a team string, treat as possible ML
    return True

s["_possible_ml"] = s["side"].apply(is_possible_ml_side)

# For the missing games, check snapshots for possible ML rows
rows = []
for gid in missing["game_id"].unique().tolist():
    ss = s[s["game_id"]==gid]
    rows.append({
        "game_id": gid,
        "game": ss["game"].iloc[0] if len(ss) else "",
        "snap_rows": len(ss),
        "possible_ml_rows": int(ss["_possible_ml"].sum()),
        "unique_sides": ss["side"].nunique()
    })

out = pd.DataFrame(rows).sort_values(["possible_ml_rows","snap_rows"], ascending=[True, False])
print("\n[CHECK SNAPSHOTS FOR ML-LIKE ROWS]")
print(out.head(40).to_string(index=False))

print("\nIf possible_ml_rows == 0 for most: DK splits feed is not providing ML for those games.")
print("If possible_ml_rows > 0 but dashboard ML is blank: classification/pivot bug in build_dashboard.")
