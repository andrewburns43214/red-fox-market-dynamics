import pandas as pd

resolved = pd.read_csv("data/results_resolved.csv", dtype=str)

print("Rows with actual scores:")
mask = (
    resolved["team1_score"].notna() &
    resolved["team2_score"].notna()
)

print(mask.sum())
