import pandas as pd

d = pd.read_csv("data/dashboard.csv")

score_cols = [c for c in d.columns if c.endswith("_model_score")]

print("\nScore column stats:\n")
for c in score_cols:
    s = pd.to_numeric(d[c], errors="coerce")
    print(f"{c:25} min={s.min():5.1f}  avg={s.mean():5.1f}  max={s.max():5.1f}")

print("\nTop strongest signals:\n")
print(d.sort_values(score_cols[0], ascending=False)[score_cols + ["sport","game"]].head(15))
