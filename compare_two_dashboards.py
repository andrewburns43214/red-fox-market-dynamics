import pandas as pd

A = pd.read_csv(r".\\bak\\BAKEOFF_YYYYMMDD_HHMMSS\\dashboard_main.py.A.csv", keep_default_na=False, dtype=str)
B = pd.read_csv(r".\\bak\\BAKEOFF_YYYYMMDD_HHMMSS\\dashboard_main.py.B.csv", keep_default_na=False, dtype=str)

keys = ["sport","game_id","game"]
for k in keys:
    if k not in A.columns or k not in B.columns:
        raise SystemExit(f"Missing key column: {k}")

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

score_cols = [c for c in A.columns if c.endswith("_model_score") and c in B.columns]
if not score_cols:
    raise SystemExit("No shared *_model_score columns to compare.")

m = A.merge(B, on=keys, how="inner", suffixes=("_A","_B"))
print("ROWS_MERGED=", len(m))
print("SCORE_COLS=", score_cols)

for c in score_cols:
    da = to_num(m[c+"_A"])
    db = to_num(m[c+"_B"])
    diff = (da - db).abs()
    print(f"\n{c}: mean_abs_diff={diff.mean():.3f} max_abs_diff={diff.max()} changed_ge1={(diff>=1).sum()} changed_ge3={(diff>=3).sum()}")
    # show top 10 diffs
    top = m.loc[diff.sort_values(ascending=False).head(10).index, keys + [c+"_A", c+"_B"]]
    print(top.to_string(index=False))
