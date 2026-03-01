import pandas as pd

rs = pd.read_csv("data/row_state.csv")

# sort by most recent update
rs = rs.sort_values("last_ts", ascending=False)

print(rs[[
    "sport",
    "game_id",
    "market",
    "side",
    "last_score",
    "last_net_edge",
    "last_bucket",
    "timing_bucket",
    "last_ts"
]].head(15))
