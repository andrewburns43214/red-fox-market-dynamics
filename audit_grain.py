import pandas as pd
from pathlib import Path

CANON_KEYS = ["sport","game_id","market_display","side"]
AGG_KEYS   = ["sport","game_id","market_display"]

def check(path, keys):
  print("\n" + "-"*90)
  print(path, "keys=", keys)
  if not Path(path).exists():
    print("MISSING"); return
  df = pd.read_csv(path)
  missing = [k for k in keys if k not in df.columns]
  if missing:
    print("MISSING KEYS:", missing); return
  dup = df.duplicated(keys).sum()
  print("rows:", len(df), "dups:", int(dup))
  # show a few duplicates if any
  if dup:
    print("SAMPLE DUP KEYS:")
    d = df[df.duplicated(keys, keep=False)][keys].head(10)
    print(d.to_string(index=False))

check("data/snapshots.csv", CANON_KEYS)         # should be canonical-ish (raw feed + injectors)
check("data/results_resolved.csv", CANON_KEYS)  # MUST be canonical grain
check("data/row_state.csv", CANON_KEYS)         # SHOULD be canonical grain (state per row)
check("data/signal_ledger.csv", CANON_KEYS)     # typically canonical grain (event log per row)
check("data/dashboard.csv", AGG_KEYS)           # MUST be aggregated grain
