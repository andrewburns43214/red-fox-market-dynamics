import pandas as pd
l = pd.read_csv("data/signal_ledger.csv", nrows=5)
print("\nCOLUMNS:\n", list(l.columns))
