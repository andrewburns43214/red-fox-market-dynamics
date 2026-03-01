import pandas as pd

d = pd.read_csv("data/row_state.csv", keep_default_na=False, dtype=str)

def parse(col):
    return pd.to_datetime(col, errors="coerce", utc=True)

ts = parse(d["ts"])
peak = parse(d["peak_ts"])

backwards = (peak > ts).sum()
missing_peak = peak.isna().sum()
missing_ts = ts.isna().sum()

print("ROWS:", len(d))
print("PEAK AFTER CURRENT (should be 0):", backwards)
print("MISSING peak_ts:", missing_peak)
print("MISSING ts:", missing_ts)
