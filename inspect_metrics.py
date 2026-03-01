import pandas as pd

print("ROW_STATE")
rs = pd.read_csv("data/row_state.csv")
print("Rows:", len(rs))
print(rs.head())

print("\nSIGNAL_LEDGER")
sl = pd.read_csv("data/signal_ledger.csv")
print("Rows:", len(sl))
print(sl.head())
