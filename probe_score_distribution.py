import pandas as pd
d = pd.read_csv("data/dashboard.csv", dtype=str)

scores = d["max_side_score"].astype(float)

print("Max score:", scores.max())
print("Top 15 scores:", sorted(scores.unique(), reverse=True)[:15])
print("Count >=72:", (scores >= 72).sum())
