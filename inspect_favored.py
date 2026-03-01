import pandas as pd

dashboard = pd.read_csv("data/dashboard.csv")
results = pd.read_csv("data/results_resolved.csv")

print("Dashboard favored_side sample:")
print(dashboard["favored_side"].dropna().unique()[:10])

print("\nResults side sample:")
print(results["side"].dropna().unique()[:10])
