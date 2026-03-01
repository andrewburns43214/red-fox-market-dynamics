import pandas as pd
d = pd.read_csv("data/dashboard.csv", keep_default_na=False, dtype=str)

print("\n[TIMING BUCKET]")
if "timing_bucket" not in d.columns:
    print("missing timing_bucket"); raise SystemExit(1)

vals = d["timing_bucket"].fillna("").astype(str).str.strip().value_counts(dropna=False)
print(vals.to_string())

bad = d[~d["timing_bucket"].isin(["EARLY","MID","LATE",""])]
print("\nbad rows:", len(bad))
if len(bad):
    print(bad[["game","dk_start_iso","timing_bucket"]].head(20).to_string(index=False))
