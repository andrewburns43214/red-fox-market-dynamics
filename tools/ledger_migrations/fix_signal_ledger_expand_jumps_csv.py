import csv
from pathlib import Path

INP = Path(r"data/signal_ledger.csv")
OUT = Path(r"data/signal_ledger.csv")  # overwrite in place

rank = {"NO_BET": 0, "LEAN": 1, "BET": 2, "STRONG_BET": 3}
order = ["NO_BET", "LEAN", "BET", "STRONG_BET"]

def norm_bucket(x: str) -> str:
    x = (x or "").strip().upper()
    if x not in rank:
        return "NO_BET"
    return x

rows_out = []

with INP.open(newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    fieldnames = list(r.fieldnames or [])
    if not fieldnames:
        raise SystemExit("[fix] FAIL: signal_ledger.csv has no header / no columns")

    for row in r:
        ev = (row.get("event") or "").strip()
        if ev != "THRESHOLD_CROSS":
            rows_out.append(row)
            continue

        a = norm_bucket(row.get("from_bucket"))
        b = norm_bucket(row.get("to_bucket"))

        i0 = rank.get(a, 0)
        i1 = rank.get(b, 0)

        # adjacent or same -> keep as-is (but normalized)
        if abs(i1 - i0) <= 1:
            row["from_bucket"] = a
            row["to_bucket"] = b
            rows_out.append(row)
            continue

        # expand jumps into adjacent steps
        if i1 > i0:
            path = order[i0:i1+1]             # e.g. NO_BET->LEAN->BET
        else:
            path = list(reversed(order[i1:i0+1]))  # e.g. STRONG_BET->BET->LEAN

        base = dict(row)
        for x, y in zip(path[:-1], path[1:]):
            r2 = dict(base)
            r2["from_bucket"] = x
            r2["to_bucket"] = y
            rows_out.append(r2)

# write back
tmp = OUT.with_suffix(".tmp")
with tmp.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for row in rows_out:
        # guarantee all columns exist
        for c in fieldnames:
            if c not in row or row[c] is None:
                row[c] = ""
        w.writerow(row)

tmp.replace(OUT)

print(f"[fix] OK: wrote {OUT} rows={len(rows_out)}")
