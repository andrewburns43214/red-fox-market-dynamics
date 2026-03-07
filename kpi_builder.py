import pandas as pd
from pathlib import Path

# -------------------------------------------------
# LOAD
# -------------------------------------------------
results_path = Path("data/results_resolved.csv")
state_path = Path("data/row_state.csv")

if not results_path.exists():
    raise SystemExit("results_resolved.csv not found")

if not state_path.exists():
    raise SystemExit("row_state.csv not found")

r = pd.read_csv(results_path)
s = pd.read_csv(state_path)

print("[kpi] total rows in results_resolved:", len(r))

# -------------------------------------------------
# PREFLIGHT VALIDATION
# -------------------------------------------------
resolved = r[r["result"].isin(["WIN","LOSS","PUSH"])].copy()
graded = r[r["result"].isin(["WIN","LOSS"])].copy()

print("[kpi] resolved rows (incl PUSH):", len(resolved))
print("[kpi] graded rows (WIN/LOSS only):", len(graded))

print("[kpi] NaN game_decision in graded:",
      graded["game_decision"].isna().sum())

print("[kpi] NaN net_edge in graded:",
      graded["net_edge"].isna().sum())

if graded["game_decision"].isna().any():
    raise SystemExit("Frozen game_decision missing — aborting")

if graded["net_edge"].isna().any():
    raise SystemExit("Frozen net_edge missing — aborting")

# -------------------------------------------------
# CANONICAL JOIN
# -------------------------------------------------
k = graded.merge(
    s[["sport","game_id","market_display","side","last_score","peak_score"]],
    on=["sport","game_id","market_display","side"],
    how="left"
)

if k["last_score"].isna().any():
    raise SystemExit("Missing last_score after canonical join — aborting")

print("[kpi] rows after canonical join:", len(k))

# -------------------------------------------------
# SUMMARY KPIs
# -------------------------------------------------
summary_rows = []

overall_wr = (k["result"] == "WIN").mean()
summary_rows.append({
    "metric":"overall_win_rate",
    "value":round(overall_wr,4),
    "n":len(k)
})

for d, g in k.groupby("game_decision"):
    summary_rows.append({
        "metric":f"decision_{d}_win_rate",
        "value":round((g["result"]=="WIN").mean(),4),
        "n":len(g)
    })

for sp, g in k.groupby("sport"):
    summary_rows.append({
        "metric":f"sport_{sp}_win_rate",
        "value":round((g["result"]=="WIN").mean(),4),
        "n":len(g)
    })

for m, g in k.groupby("market_display"):
    summary_rows.append({
        "metric":f"market_{m}_win_rate",
        "value":round((g["result"]=="WIN").mean(),4),
        "n":len(g)
    })

pd.DataFrame(summary_rows).to_csv("data/kpi_summary.csv", index=False)

# -------------------------------------------------
# LOCKED BUCKET DEFINITIONS
# -------------------------------------------------

def score_bucket(x):
    if x < 60: return "<60"
    if x <= 66: return "60-66"
    if x <= 69: return "67-69"
    if x <= 74: return "70-74"
    return "75+"

def edge_bucket(x):
    if x <= 4: return "0-4"
    if x <= 8: return "5-8"
    if x <= 12: return "9-12"
    return "13+"

k["score_bucket"] = k["last_score"].apply(score_bucket)
k["edge_bucket"] = k["net_edge"].apply(edge_bucket)

bucket_rows = []

for b, g in k.groupby("score_bucket"):
    bucket_rows.append({
        "bucket_type":"score_last",
        "bucket":b,
        "win_rate":round((g["result"]=="WIN").mean(),4),
        "n":len(g)
    })

for b, g in k.groupby("edge_bucket"):
    bucket_rows.append({
        "bucket_type":"net_edge",
        "bucket":b,
        "win_rate":round((g["result"]=="WIN").mean(),4),
        "n":len(g)
    })

pd.DataFrame(bucket_rows).to_csv("data/kpi_bucket_summary.csv", index=False)

# -------------------------------------------------
# CLV ANALYSIS (v2.1)
# -------------------------------------------------
if "clv" in k.columns:
    k["clv_num"] = pd.to_numeric(k["clv"], errors="coerce")
    _clv_valid = k[k["clv_num"].notna()].copy()

    if len(_clv_valid) > 0:
        # Average CLV overall
        summary_rows.append({
            "metric": "avg_clv",
            "value": round(_clv_valid["clv_num"].mean(), 4),
            "n": len(_clv_valid)
        })

        # Average CLV by decision level
        for d, g in _clv_valid.groupby("game_decision"):
            if len(g) >= 2:
                summary_rows.append({
                    "metric": f"avg_clv_{d}",
                    "value": round(g["clv_num"].mean(), 4),
                    "n": len(g)
                })

        # Average CLV by sport
        for sp, g in _clv_valid.groupby("sport"):
            if len(g) >= 2:
                summary_rows.append({
                    "metric": f"avg_clv_sport_{sp}",
                    "value": round(g["clv_num"].mean(), 4),
                    "n": len(g)
                })

        # Average CLV by market
        for m, g in _clv_valid.groupby("market_display"):
            if len(g) >= 2:
                summary_rows.append({
                    "metric": f"avg_clv_market_{m}",
                    "value": round(g["clv_num"].mean(), 4),
                    "n": len(g)
                })

        # CLV buckets
        def clv_bucket(x):
            if x < -2: return "<-2"
            if x < 0: return "-2 to 0"
            if x <= 1: return "0 to +1"
            if x <= 3: return "+1 to +3"
            return "3+"

        _clv_valid["clv_bucket"] = _clv_valid["clv_num"].apply(clv_bucket)
        for b, g in _clv_valid.groupby("clv_bucket"):
            bucket_rows.append({
                "bucket_type": "clv",
                "bucket": b,
                "win_rate": round((g["result"] == "WIN").mean(), 4),
                "n": len(g)
            })

        # Re-write with CLV data included
        pd.DataFrame(summary_rows).to_csv("data/kpi_summary.csv", index=False)
        pd.DataFrame(bucket_rows).to_csv("data/kpi_bucket_summary.csv", index=False)
        print(f"[kpi] CLV analysis: {len(_clv_valid)} rows, avg CLV = {_clv_valid['clv_num'].mean():.4f}")
    else:
        print("[kpi] no valid CLV data yet (pre-v2.1 decisions)")
else:
    print("[kpi] CLV column not in results (pre-v2.1)")

print("[kpi] build complete.")
print("[kpi] summary rows:", len(summary_rows))
print("[kpi] bucket rows:", len(bucket_rows))
