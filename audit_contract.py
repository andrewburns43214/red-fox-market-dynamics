import csv
from pathlib import Path

def read_csv(path):
    p = Path(path)
    if not p.exists():
        print(f"FAIL: missing {path}")
        return None, None
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    return r.fieldnames, rows

cols, rows = read_csv("data/dashboard.csv")
if cols is None:
    raise SystemExit(2)

print("=== CONTRACT CHECKS (dashboard.csv) ===")

# row_status required eventually per spec; check presence now
print("has_row_status:", "row_status" in cols)

# If present, verify values are only ACTIVE/SCHEDULED/INVALID
if "row_status" in cols:
    vals = sorted({(r.get("row_status") or "").strip() for r in rows})
    print("row_status_unique:", vals)
    bad = [v for v in vals if v and v not in ("ACTIVE","SCHEDULED","INVALID")]
    if bad:
        print("FAIL: invalid row_status values:", bad)

# Timing bucket required eventually; check presence now
print("has_timing_bucket:", "timing_bucket" in cols)

# Core scoring outputs must exist at side-level and be carried forward
print("has_model_score:", "model_score" in cols)

# Aggregation outputs per-market should exist (in LONG canonical dashboard)
print("has_net_edge_market:", "net_edge_market" in cols)
print("has_game_confidence:", "game_confidence" in cols)
