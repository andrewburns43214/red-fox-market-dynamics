import pandas as pd
d = pd.read_csv("data/signal_ledger.csv", keep_default_na=False, dtype=str)

print("ROWS:", len(d))
print("\nCOLUMNS:")
for c in d.columns:
    print(c)

print("\nLAST 10 EVENTS:")
print(d.tail(10).to_string(index=False))
