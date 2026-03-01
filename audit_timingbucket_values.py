import pandas as pd
d = pd.read_csv("data/dashboard.csv")
print(d["timing_bucket"].fillna("").replace("nan","").value_counts())
print("\nSample same-day rows:")
x = d[d["timing_bucket"].fillna("").astype(str).str.strip()!=""].copy()
print(x[["sport","game","dk_start_iso","timing_bucket"]].head(12).to_string(index=False))
