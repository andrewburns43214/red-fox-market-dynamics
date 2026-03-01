import pandas as pd
d = pd.read_csv("data/dashboard.csv", keep_default_na=False)
print("DASH sports:", d["sport"].value_counts().to_dict())
print("Rows:", len(d))
print("Columns:", list(d.columns))
