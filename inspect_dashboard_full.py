import pandas as pd
d = pd.read_csv("data/dashboard.csv", dtype=str)

print("COLUMNS:")
for c in d.columns:
    print(c)

print("\nSAMPLE model_score values:")
if "model_score" in d.columns:
    print(d["model_score"].head(10).tolist())
else:
    print("model_score column missing")
