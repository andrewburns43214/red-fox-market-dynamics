import pandas as pd

row = pd.read_csv("data/row_state.csv")
res = pd.read_csv("data/results_resolved.csv")

print("Row_state sport unique:", row["sport"].unique()[:5])
print("Resolved sport unique:", res["sport"].unique()[:5])

print("Row_state game_id dtype:", row["game_id"].dtype)
print("Resolved game_id dtype:", res["game_id"].dtype)

print("Row_state side sample:", row["side"].dropna().unique()[:5])
print("Resolved side sample:", res["side"].dropna().unique()[:5])
