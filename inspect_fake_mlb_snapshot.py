import pandas as pd

s = pd.read_csv("data/snapshots.csv", keep_default_na=False, dtype=str)

mlb = s[s["sport"].str.upper()=="MLB"].copy()

print("TOTAL MLB SNAPSHOT ROWS:", len(mlb))

cols = [c for c in [
    "sport","game","market","side",
    "dk_start_iso","timestamp",
    "current_line","current_odds"
] if c in mlb.columns]

print("\n--- RAW MLB SNAPSHOT SAMPLE ---")
print(mlb[cols].head(20).to_string(index=False))
