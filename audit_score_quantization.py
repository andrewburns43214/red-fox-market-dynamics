import pandas as pd

d = pd.read_csv("data/dashboard.csv")

score_cols = [c for c in d.columns if c.endswith("_model_score")]
print("score_cols:", score_cols)

for c in score_cols:
    x = pd.to_numeric(d[c], errors="coerce")
    print("\n", c)
    print("  count:", x.notna().sum())
    print("  min/max:", float(x.min()), float(x.max()))
    print("  unique (top 15):")
    print(x.value_counts().head(15))

print("\nTiming bucket counts:")
print(d["timing_bucket"].fillna("").replace("nan","").value_counts())
