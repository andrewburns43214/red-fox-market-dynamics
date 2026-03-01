import pandas as pd

rs = pd.read_csv("data/results_resolved.csv", dtype=str)
print("Resolved rows:", len(rs))

print("Sample:")
print(rs.head(10))
