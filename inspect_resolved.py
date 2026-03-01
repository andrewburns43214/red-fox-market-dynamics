import pandas as pd

resolved = pd.read_csv("data/results_resolved.csv", dtype=str)

print("Total resolved rows:", len(resolved))
print("\nColumns:")
print(resolved.columns.tolist())

print("\nSample rows:")
print(resolved.head(10))

print("\nUnique game_id count:", resolved["game_id"].nunique())
