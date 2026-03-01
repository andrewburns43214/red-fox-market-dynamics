import pandas as pd
import numpy as np

PATH = "data/dashboard.csv"
d = pd.read_csv(PATH)

print("DASHBOARD ROWS:", len(d))
print("COLUMNS:", len(d.columns))

# --- helper: find likely column names safely (we won’t assume exact casing) ---
cols = {c.lower(): c for c in d.columns}

def pick(*names):
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None

sport_c = pick("sport")
gid_c = pick("game_id")
mkt_c = pick("market_display")
conf_c = pick("game_confidence", "confidence", "game_conf")
min_c  = pick("min_side_score", "min_score")
max_c  = pick("max_side_score", "max_score")
edge_c = pick("net_edge_market", "net_edge")
dec_c  = pick("decision", "game_decision")
fav_c  = pick("favored_side", "favored", "pick", "pick_side")

print("\nFOUND COLS:")
print(" sport:", sport_c)
print(" game_id:", gid_c)
print(" market_display:", mkt_c)
print(" game_confidence:", conf_c)
print(" min_side_score:", min_c)
print(" max_side_score:", max_c)
print(" net_edge:", edge_c)
print(" decision:", dec_c)
print(" favored_side:", fav_c)

need = [sport_c, gid_c, mkt_c, conf_c, min_c, max_c, edge_c, dec_c]
missing = [x for x in need if x is None]
if missing:
    raise SystemExit(f"\nFATAL: Missing required columns for audit: {missing}\n")

# numeric coercion
for c in [conf_c, min_c, max_c, edge_c]:
    d[c] = pd.to_numeric(d[c], errors="coerce")

# ---- HARD INVARIANTS ----
# 1) game_confidence == max_side_score
eps = 0.11  # tolerate 0.1 rounding artifacts
bad_conf = d[(d[conf_c].isna()) | (d[max_c].isna()) | (np.abs(d[conf_c] - d[max_c]) > eps)]

# 2) net_edge == max - min
calc_edge = d[max_c] - d[min_c]
bad_edge = d[(d[edge_c].isna()) | (d[min_c].isna()) | (d[max_c].isna()) | (np.abs(d[edge_c] - calc_edge) > eps)]

print("\n=== HARD INVARIANTS ===")
print("A) game_confidence != max_side_score:", len(bad_conf))
print("B) net_edge != max-min:", len(bad_edge))

def show(df, title, n=15):
    if len(df) == 0:
        return
    keep = [sport_c, gid_c, mkt_c, conf_c, min_c, max_c, edge_c, dec_c]
    if fav_c: keep.append(fav_c)
    print(f"\n--- {title} (showing up to {n}) ---")
    print(df[keep].head(n).to_string(index=False))

show(bad_conf, "BAD: confidence mismatch")
show(bad_edge, "BAD: edge mismatch")

# ---- DECISION DISTRIBUTIONS (no assumptions yet) ----
print("\n=== DECISION COUNTS (overall) ===")
print(d[dec_c].value_counts(dropna=False).to_string())

print("\n=== DECISION COUNTS by sport ===")
print(d.groupby(sport_c)[dec_c].value_counts(dropna=False).to_string())

# ---- SCORE/EDGE SHAPE CHECKS ----
# Potential anomaly buckets: high confidence but tiny edge, or big edge but low confidence.
hi_conf_lo_edge = d[(d[conf_c] >= 68) & (d[edge_c] <= 4.0)]
hi_conf_zero_edge = d[(d[conf_c] >= 68) & (d[edge_c] <= eps)]
big_edge_low_conf = d[(d[edge_c] >= 13) & (d[conf_c] < 60)]

print("\n=== SHAPE FLAGS ===")
print("High confidence (>=68) with low edge (<=4):", len(hi_conf_lo_edge))
print("High confidence (>=68) with ~zero edge:", len(hi_conf_zero_edge))
print("Big edge (>=13) with low confidence (<60):", len(big_edge_low_conf))

show(hi_conf_zero_edge, "FLAG: hi_conf + ~zero edge")
show(hi_conf_lo_edge, "FLAG: hi_conf + low edge")
show(big_edge_low_conf, "FLAG: big edge + low confidence")

# ---- 72.0 CLUSTER AUDIT (esp NCAAB) ----
exact_72 = d[np.isclose(d[conf_c], 72.0, atol=0.05)]
print("\n=== 72.0 CLUSTER ===")
print("Rows with game_confidence ~72.0:", len(exact_72))

if len(exact_72):
    print("\n72.0 rows by sport:")
    print(exact_72[sport_c].value_counts().to_string())

    # within NCAAB, see if edge is also clustered (e.g., 13.0)
    ncaab_72 = exact_72[exact_72[sport_c].astype(str).str.upper().str.contains("NCAAB")]
    if len(ncaab_72):
        print("\nNCAAB 72.0 net_edge value counts:")
        print(ncaab_72[edge_c].round(2).value_counts().head(20).to_string())
        print("\nNCAAB 72.0 max_side_score value counts:")
        print(ncaab_72[max_c].round(2).value_counts().head(20).to_string())
        print("\nNCAAB 72.0 min_side_score value counts:")
        print(ncaab_72[min_c].round(2).value_counts().head(20).to_string())

# ---- TOPS PER SPORT (to compare to your observed “Top 10”) ----
print("\n=== TOP 10 PER SPORT (by game_confidence desc) ===")
for sp, g in d.groupby(sport_c):
    top = g.sort_values(conf_c, ascending=False).head(10)
    keep = [sport_c, gid_c, mkt_c, conf_c, edge_c, min_c, max_c, dec_c]
    if fav_c: keep.append(fav_c)
    print(f"\n--- {sp} ---")
    print(top[keep].to_string(index=False))

print("\nDONE.")
