import pandas as pd
sn = pd.read_csv("data/snapshots.csv", dtype=str)
cols = sn.columns.tolist()
print("snapshots.csv cols:", len(cols))
print("has model_score?", "model_score" in cols)
print("has timing_bucket?", "timing_bucket" in cols)
print("has row_status?", "row_status" in cols)
print("has finals?", ("final_score_for" in cols) and ("final_score_against" in cols))
