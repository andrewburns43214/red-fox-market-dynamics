import pandas as pd
d = pd.read_csv("data/dashboard.csv", keep_default_na=False)
print("rows:", len(d))
print("cols:", len(d.columns))
print("sample cols:", list(d.columns)[:40])
print("decision cols:", [c for c in d.columns if "decision" in c.lower()])
print("score cols:", [c for c in d.columns if "model_score" in c.lower() or "score" in c.lower() or "confidence" in c.lower()])
