import pandas as pd

sl = pd.read_csv("data/signal_ledger.csv")

print("SIGNAL_LEDGER COLUMNS:")
for c in sl.columns:
    print(" -", c)

print("\nColumns containing 'edge':")
print([c for c in sl.columns if "edge" in c.lower()])
