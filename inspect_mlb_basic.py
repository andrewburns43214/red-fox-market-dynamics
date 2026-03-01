import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False, dtype=str)

print("TOTAL ROWS:", len(d))

if "sport" not in d.columns:
    raise SystemExit("No sport column found")

mlb = d[d["sport"].str.upper() == "MLB"].copy()
print("MLB ROWS:", len(mlb))

print("\n--- BASIC SAMPLE ---")
cols = [c for c in [
    "sport","game","market_display",
    "market_read","market_favors","market_why",
    "strong_eligible","strong_block_reason"
] if c in mlb.columns]

print(mlb[cols].head(20).to_string(index=False))
