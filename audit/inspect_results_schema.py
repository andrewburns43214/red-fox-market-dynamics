import pandas as pd

r = pd.read_csv("data/results_resolved.csv")

print("\n[RESULTS_RESOLVED COLUMNS]")
print(list(r.columns))

print("\nFirst 3 rows:")
print(r.head(3))
