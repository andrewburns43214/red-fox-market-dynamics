import pandas as pd

# load raw snapshot and dashboard
snap = pd.read_csv("data/snapshots.csv")
dash = pd.read_csv("data/dashboard.csv")

print("Dashboard sample:\n")
print(dash[["game","SPREAD_model_score","MONEYLINE_model_score","TOTAL_model_score","net_edge"]].head(5))

print("\nUnique net edges:", dash["net_edge"].unique())
