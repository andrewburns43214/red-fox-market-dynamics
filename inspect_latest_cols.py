import pandas as pd
import main

df = pd.read_csv("data/snapshots.csv", dtype=str)
latest = main.build_latest(df)

print("\nCOLUMNS IN LATEST:")
print(sorted(latest.columns.tolist()))
