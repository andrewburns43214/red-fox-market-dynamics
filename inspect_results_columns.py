import pandas as pd

r = pd.read_csv("data/results_resolved.csv")

print("Columns in results_resolved:")
for c in r.columns:
    print("-", c)

print("\nAny team1_score column present?", "team1_score" in r.columns)
print("Any team2_score column present?", "team2_score" in r.columns)
