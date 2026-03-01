import pandas as pd
d = pd.read_csv("data/dashboard.csv")
print("\nNET EDGE COLS:")
print([c for c in d.columns if "net_edge" in c])
print("\nSCORE COLS:")
print([c for c in d.columns if c.endswith("_model_score")])
