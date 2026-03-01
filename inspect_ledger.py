import pandas as pd
d = pd.read_csv("data/decision_freeze_ledger.csv")
print("Ledger rows:", len(d))
print(d.head())
