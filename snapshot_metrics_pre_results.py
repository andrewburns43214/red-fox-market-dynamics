import pandas as pd

# Read current metrics
df_state = pd.read_csv("data/row_state.csv", dtype=str)
df_ledger = pd.read_csv("data/signal_ledger.csv", dtype=str)

# Save pre-results snapshot
df_state.to_csv("data/row_state_pre_results.csv", index=False)
df_ledger.to_csv("data/signal_ledger_pre_results.csv", index=False)

print(f"Pre-results row_state rows: {len(df_state)}")
print(f"Pre-results signal_ledger rows: {len(df_ledger)}")
