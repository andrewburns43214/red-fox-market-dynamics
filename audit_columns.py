import pandas as pd
from pathlib import Path

paths = [
  "data/snapshots.csv",
  "data/dashboard.csv",
  "data/row_state.csv",
  "data/signal_ledger.csv",
  "data/results_resolved.csv",
  "data/final_scores_history.csv",
]

for p in paths:
  fp = Path(p)
  print("\n" + "="*90)
  print(p)
  if not fp.exists():
    print("  MISSING")
    continue

  df = pd.read_csv(fp)
  print("  rows:", len(df), " cols:", len(df.columns))
  print("  columns:")
  for c in df.columns:
    print("   -", c)
