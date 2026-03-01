import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("\nCounts >= 72")
for c in d.columns:
    if "model_score" in c:
        print(c, (d[c] >= 72).sum())
