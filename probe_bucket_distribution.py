import pandas as pd

d = pd.read_csv("data/dashboard.csv", dtype=str, keep_default_na=False)

# pick the score column metrics is using
score_col = "model_score" if "model_score" in d.columns else ("score" if "score" in d.columns else None)
print("score_col:", score_col)

if score_col:
    s = pd.to_numeric(d[score_col], errors="coerce").fillna(0)
    print("min/max:", float(s.min()), float(s.max()))
    print(">=60:", int((s>=60).sum()), ">=68:", int((s>=68).sum()), ">=72:", int((s>=72).sum()))
else:
    print("no score col found")
