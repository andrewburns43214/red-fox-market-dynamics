import pandas as pd

d = pd.read_csv("data/row_state.csv", keep_default_na=False, dtype=str)

needed = [
"sport","game_id","market","side","logic_version","ts",
"score","net_edge","timing_bucket","last_decision",
"strong_precheck","strong_certified","strong_block_reasons",
"strong72_now","strong_streak","peak_score","last_score"
]

print("\nMISSING COLUMNS:")
for c in needed:
    if c not in d.columns:
        print(" -", c)

print("\nALL COLUMNS:")
for c in d.columns:
    print(c)
