import pandas as pd

d = pd.read_csv("data/dashboard.csv", dtype=str, keep_default_na=False)
print("dashboard sports:", sorted(set(d["sport"].astype(str).str.lower())))

# if MLB is absent from dashboard, we need to inspect latest earlier in report.
print("NOTE: If mlb missing here, it died before game_view.")
