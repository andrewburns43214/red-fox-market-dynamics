import pandas as pd

df = pd.read_csv("data/row_state.csv", dtype=str)

df["last_score"] = pd.to_numeric(df["last_score"], errors="coerce")

row = df.sort_values("last_score", ascending=False).head(1)
print(row.T)
