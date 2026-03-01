import pandas as pd

led = pd.read_csv("data/signal_ledger.csv", dtype=str)
print("ledger rows:", len(led))
print("columns:", led.columns.tolist())

print("\nSAMPLE games:")
print(led["game"].dropna().head(10).to_string(index=False))

print("\nTO_BUCKET counts:")
print(led["to_bucket"].fillna("").value_counts().head(20).to_string())
