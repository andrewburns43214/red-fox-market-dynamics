import pandas as pd
df = pd.read_csv("data/row_state.csv", dtype=str)

df["last_score"] = pd.to_numeric(df["last_score"], errors="coerce")

cur = df[df["last_seen_tick"] == df["last_seen_tick"].max()].copy()

print("Fresh rows:", len(cur))
print("Max score fresh:", cur["last_score"].max())
print("Count >=72 fresh:", (cur["last_score"] >= 72).sum())
print("Strong now true count:", (cur["strong72_now"] == "True").sum())
print("Certified true count:", (cur["strong_certified"] == "True").sum())
