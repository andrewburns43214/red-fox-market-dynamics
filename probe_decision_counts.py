import pandas as pd
d = pd.read_csv("data/dashboard.csv", dtype=str)

if "decision" in d.columns:
    print(d["decision"].value_counts())
else:
    print("No decision column found")
