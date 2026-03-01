import pandas as pd

state = pd.read_csv("data/row_state.csv", keep_default_na=False)
gid = "33688450"

g = state[state["game_id"].astype(str)==gid].copy()
g = g[["market","side","last_score","last_net_edge"]]
print(g.sort_values(["market","last_score"], ascending=[True,False]).to_string(index=False))
