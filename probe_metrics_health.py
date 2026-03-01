import pandas as pd

s = pd.read_csv("data/row_state.csv", dtype=str, keep_default_na=False)
l = pd.read_csv("data/signal_ledger.csv", dtype=str, keep_default_na=False)

print("row_state rows:", len(s))
print("unique rows:", s[['sport','game_id','market','side']].drop_duplicates().shape[0])
print("ledger rows:", len(l))

# detect churn (rows rewritten each run)
dup = s[['sport','game_id','market','side']].duplicated().sum()
print("duplicate identity rows:", dup)
