import pandas as pd

s = pd.read_csv("data/row_state.csv", dtype=str, keep_default_na=False)

print("\nROW STATE SAMPLE\n")
print(s[['sport','game_id','market','side','score','timing_bucket']].head(20))
