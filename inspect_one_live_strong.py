import pandas as pd

state = pd.read_csv("data/row_state.csv", keep_default_na=False)

gid = "33688447"

g = state[state["game_id"].astype(str)==gid].copy()

print(g[[
    "market","side","last_score","last_net_edge",
    "strong_streak","strong72_streak","strong72_now",
    "timing_bucket","strong_block_reasons",
    "strong_precheck","strong_certified"
]].to_string(index=False))
