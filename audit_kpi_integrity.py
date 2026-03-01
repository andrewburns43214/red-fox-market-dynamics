import pandas as pd

r = pd.read_csv("data/results_resolved.csv")
s = pd.read_csv("data/row_state.csv")

print("results_resolved rows:", len(r))
print("row_state rows:", len(s))

print("Result values:", r["result"].dropna().unique())
print("Decision nulls where result exists:",
      r[(r["result"].notna()) & (r["game_decision"].isna())].shape[0])

dup = r.duplicated(subset=["sport","game_id","market_display","side"]).sum()
print("Duplicate canonical rows (results_resolved):", dup)

dup2 = s.duplicated(subset=["sport","game_id","side"]).sum()
print("Duplicate key rows (row_state join grain):", dup2)
