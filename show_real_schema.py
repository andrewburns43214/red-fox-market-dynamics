import pandas as pd

real = pd.read_csv("data/snapshots.REAL_BACKUP.csv", dtype=str)

print("Columns:", list(real.columns))
print("\nExample row:\n")
print(real.iloc[0])
