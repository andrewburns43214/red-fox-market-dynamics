import csv
from collections import Counter

LEDGER = r"data/signal_ledger.csv"

ALLOWED = {
    ("NO_BET","LEAN"),("LEAN","BET"),("BET","STRONG_BET"),
    ("STRONG_BET","BET"),("BET","LEAN"),("LEAN","NO_BET")
}

def main():
    with open(LEDGER, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("[ledger_check] FAIL: empty ledger")

    # logic_version check
    lv = Counter((r.get("logic_version") or "").strip() for r in rows)
    if "" in lv:
        print("[ledger_check] WARN: blank logic_version rows:", lv[""])
    if "v?" in lv or "V?" in lv:
        raise SystemExit(f"[ledger_check] FAIL: found v? logic_version rows: {lv}")

    # adjacency check (THRESHOLD_CROSS only)
    bad = []
    for r in rows:
        if (r.get("event") or "").strip() != "THRESHOLD_CROSS":
            continue
        a = (r.get("from_bucket") or "").strip().upper()
        b = (r.get("to_bucket") or "").strip().upper()
        if (a,b) not in ALLOWED:
            bad.append((a,b,r.get("ts"),r.get("sport"),r.get("game_id"),r.get("market"),r.get("side")))

    if bad:
        print("[ledger_check] BAD transitions (first 25):")
        for x in bad[:25]:
            print("  ", x)
        raise SystemExit(f"[ledger_check] FAIL: bad_transition_count={len(bad)}")

    # tail sanity (last 25 must be v1.1 if that’s your active version)
    tail = rows[-25:]
    tail_bad = [r for r in tail if (r.get("logic_version") or "").strip() != "v1.1"]
    if tail_bad:
        print("[ledger_check] FAIL: tail has non-v1.1 rows, example:", tail_bad[0])
        raise SystemExit("[ledger_check] FAIL: tail_bad_count>0")

    print("[ledger_check] OK",
          "rows=", len(rows),
          "logic_version_counts=", dict(lv),
          "bad_transition_count=0",
          "tail_bad_count=0")

if __name__ == "__main__":
    main()
