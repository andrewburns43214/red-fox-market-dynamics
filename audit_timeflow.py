import pandas as pd

d = pd.read_csv("data/row_state.csv", keep_default_na=False, dtype=str)

ts = pd.to_datetime(d["ts"], errors="coerce")
peak = pd.to_datetime(d["peak_ts"], errors="coerce")

backwards = (peak > ts).sum()

print("ROWS:", len(d))
print("PEAK AFTER CURRENT (should be 0):", backwards)

missing = d["ts"].str.strip().eq("").sum()
print("MISSING TS:", missing)
