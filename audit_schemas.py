import csv, os, sys
from pathlib import Path

def head(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"__MISSING__::{path}"
    # Read first non-empty line as header
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for line in f:
            line = line.strip("\r\n")
            if line.strip():
                return line
    return "__EMPTY__"

def count_rows(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return -1
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    # rows includes header
    return max(0, len(rows)-1)

def cols_csv(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        hdr = next(r, [])
    return hdr

def main():
    files = ["data/dashboard.csv", "data/row_state.csv", "data/signal_ledger.csv"]
    print("=== FILE PRESENCE + ROW COUNTS ===")
    for fp in files:
        exists = Path(fp).exists()
        print(f"{fp}: exists={exists} rows={count_rows(fp)}")

    print("\n=== dashboard.csv COLUMNS (canonical LONG) ===")
    dash_cols = cols_csv("data/dashboard.csv")
    if dash_cols is None:
        print("FAIL: data/dashboard.csv missing")
        sys.exit(2)
    for i,c in enumerate(dash_cols):
        print(f"{i+1:03d}  {c}")

    print("\n=== row_state.csv HEADER (DB TABLE / IMMUTABLE ORDER) ===")
    print(head("data/row_state.csv"))

    print("\n=== signal_ledger.csv HEADER (DB TABLE / IMMUTABLE ORDER) ===")
    print(head("data/signal_ledger.csv"))

if __name__ == "__main__":
    main()
