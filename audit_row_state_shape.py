import pandas as pd
d = pd.read_csv("data/row_state.csv", dtype=str)
print("row_state cols has side?", "side" in d.columns)
print("unique sides:", d["side"].nunique() if "side" in d.columns else "n/a")
print("sample cols:", list(d.columns)[:30])
