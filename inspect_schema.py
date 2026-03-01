import pandas as pd

dash = pd.read_csv("data/dashboard.csv", dtype=str, keep_default_na=False)
state = pd.read_csv("data/row_state.csv", dtype=str, keep_default_na=False)
ledger = pd.read_csv("data/signal_ledger.csv", dtype=str, keep_default_na=False)

print("\n=== DASHBOARD COLUMNS ===")
print(list(dash.columns))

print("\n=== ROW_STATE COLUMNS ===")
print(list(state.columns))

print("\n=== SIGNAL_LEDGER COLUMNS ===")
print(list(ledger.columns))

print("\nROW COUNTS")
print("dashboard:", len(dash))
print("row_state:", len(state))
print("ledger:", len(ledger))
