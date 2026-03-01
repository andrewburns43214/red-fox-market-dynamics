import pandas as pd
l = pd.read_csv("data/signal_ledger.csv")

print("\nUnique event values:")
for c in l.columns:
    if "event" in c.lower() or "type" in c.lower():
        print("\nCOLUMN:", c)
        print(l[c].dropna().unique()[:50])
