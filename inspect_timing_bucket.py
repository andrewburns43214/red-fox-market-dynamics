import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)
tb = d["timing_bucket"].astype(str).str.strip()
print("timing_bucket blank rows:", (tb=="").sum(), "/", len(d))
print("timing_bucket counts:", tb.value_counts(dropna=False).to_dict())
