import pandas as pd

rs = pd.read_csv("data/row_state.csv", dtype=str)

gid = "33474156"

g = rs[rs["game_id"] == gid]

print(g[[
    "market",
    "side",
    "last_score",
    "last_bucket",
    "peak_score"
]].sort_values(["market","side"]))
