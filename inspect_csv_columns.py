from pathlib import Path
import pandas as pd

FILES = [
    "data/snapshots.csv",
    "data/dashboard.csv",
    "data/row_state.csv",
    "data/signal_ledger.csv",
]

for f in FILES:
    p = Path(f)
    print("\n=== ", f, " ===")
    if not p.exists():
        print("MISSING")
        continue
    try:
        df = pd.read_csv(p, nrows=5)
        # full columns list from header
        cols = list(df.columns)
        print("rows_sampled=5")
        print("col_count=", len(cols))
        for c in cols:
            print(" -", c)
    except Exception as e:
        print("ERROR reading:", e)
