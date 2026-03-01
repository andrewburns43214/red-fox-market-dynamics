import pandas as pd
import numpy as np

d = pd.read_csv("data/dashboard.csv")

score_cols = [c for c in d.columns if c.endswith("_model_score")]
print("SCORE COLS:", score_cols)

def step_report(col):
    s = pd.to_numeric(d[col], errors="coerce").dropna()
    if s.empty:
        return
    u = np.sort(s.unique())
    diffs = np.diff(u)
    diffs = diffs[diffs > 1e-9]
    min_step = float(diffs.min()) if len(diffs) else None

    print("\n==", col, "==")
    print("count:", len(s), "min:", float(s.min()), "max:", float(s.max()))
    print("unique:", len(u), "min_step:", min_step)
    print("top exact values:")
    vc = s.value_counts().head(12)
    print(vc.to_string())

for col in score_cols:
    step_report(col)
