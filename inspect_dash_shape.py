import pandas as pd
d = pd.read_csv("data/dashboard.csv", dtype=str, keep_default_na=False)
print("rows:", len(d))
print("market_display unique:", sorted(d["market_display"].unique().tolist()) if "market_display" in d.columns else None)
