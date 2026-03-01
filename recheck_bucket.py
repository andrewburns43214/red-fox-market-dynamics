import pandas as pd

GAME = "Utah State @ San Diego State"

state = pd.read_csv("data/row_state.csv", dtype=str)
gid = state[state["game_id"].notna()]

# find game_id from dashboard
dash = pd.read_csv("data/dashboard.csv", dtype=str)
gid = dash[dash["game"] == GAME]["game_id"].iloc[0]

rows = state[state["game_id"] == gid]

print(rows[["market","side","timing_bucket","last_score"]].to_string(index=False))
