import pandas as pd

state = pd.read_csv("data/row_state.csv", keep_default_na=False)
gid = "33694749"

g = state[state["game_id"].astype(str)==gid].copy()
print(g.sort_values(["market","last_score"], ascending=[True,False]).to_string(index=False))
