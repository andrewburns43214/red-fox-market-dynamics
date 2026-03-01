import pandas as pd

sn = pd.read_csv("data/snapshots.csv", dtype=str)

# look at rows with high score
sn["model_score"] = pd.to_numeric(sn.get("model_score", ""), errors="coerce")

high = sn[sn["model_score"] >= 72].copy()

print("High-score rows:", len(high))
print(high[["sport","game_id","market","side","model_score","timing_bucket"]].head(20))

# check for strong flags
strong_cols = [c for c in sn.columns if "strong" in c.lower()]
print("\nStrong-related columns:", strong_cols)

if strong_cols:
    print(high[strong_cols].head(10))
