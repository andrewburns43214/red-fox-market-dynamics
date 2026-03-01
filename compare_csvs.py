import pandas as pd

d13 = pd.read_csv("data/dashboard_FEB13.csv")
d23 = pd.read_csv("data/dashboard_FEB23.csv")

print("\n=== COLUMN CHECK ===")
print("FEB13 columns:", list(d13.columns))
print("FEB23 columns:", list(d23.columns))

print("\n=== SHAPE CHECK ===")
print("FEB13 shape:", d13.shape)
print("FEB23 shape:", d23.shape)

print("\n=== HEAD DIFFERENCE CHECK ===")
print(d13.head(3))
print(d23.head(3))

# Try aligning on key columns if they exist
common_cols = [c for c in ["sport","game_id","market_display"] if c in d13.columns and c in d23.columns]

if common_cols:
    merged = d13.merge(d23, on=common_cols, suffixes=("_13","_23"))
    if "game_confidence_13" in merged.columns and "game_confidence_23" in merged.columns:
        diff = (merged["game_confidence_23"] - merged["game_confidence_13"]).abs().sum()
        print("\nTotal confidence absolute diff:", diff)
