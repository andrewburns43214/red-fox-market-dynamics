import pandas as pd
d = pd.read_csv("data/row_state.csv", keep_default_na=False)

print("missing last_ts:", (d["last_ts"].astype(str).str.strip()=="").sum())
print("missing peak_ts:", (d["peak_ts"].astype(str).str.strip()=="").sum())
print(d[["last_ts","peak_ts","last_score","peak_score"]].head(10))
