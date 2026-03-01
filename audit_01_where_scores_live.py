import pandas as pd, os

paths = [
  ("row_state", "data/row_state.csv"),
  ("signal_ledger", "data/signal_ledger.csv"),
  ("engine_truth", "data/_engine_truth.csv"),
]

for name, path in paths:
    if not os.path.exists(path):
        print(f"\n{name}: MISSING ({path})")
        continue
    df = pd.read_csv(path, dtype=str)
    cols = df.columns.tolist()
    print(f"\n{name}: {path}")
    print("rows:", len(df), "| cols:", len(cols))
    # show the most relevant columns if present
    want = [c for c in cols if any(k in c.lower() for k in [
        "model_score","score","timing","bucket","strong","decision","market_read",
        "row_status","net_edge","bets","money","snapshot"
    ])]
    print("relevant cols:", want)
