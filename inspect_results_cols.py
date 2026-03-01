import pandas as pd
r = pd.read_csv("data/results_resolved.csv", dtype=str)
print(r.columns.tolist())
