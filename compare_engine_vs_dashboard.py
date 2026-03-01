import pandas as pd

# Load truth (engine state)
state = pd.read_csv("data/row_state.csv", keep_default_na=False)

# Load dashboard
dash = pd.read_csv("data/dashboard.csv", keep_default_na=False)

games = ["33469998","33471240","33471245","33471246"]

print("\n================ ENGINE (row_state) ================")
eng = state[state["game_id"].astype(str).isin(games)].copy()
eng = eng[["sport","game_id","market","side","last_score","last_net_edge"]]
eng = eng.sort_values(["game_id","market"])
print(eng.to_string(index=False))

print("\n================ DASHBOARD (WIDE) ==================")
dash_sub = dash[dash["game_id"].astype(str).isin(games)].copy()

cols = [
    "sport","game_id","game",
    "SPREAD_favored","SPREAD_model_score","SPREAD_net_edge",
    "MONEYLINE_favored","MONEYLINE_model_score","MONEYLINE_net_edge",
    "TOTAL_favored","TOTAL_model_score","TOTAL_net_edge",
    "net_edge"
]

existing = [c for c in cols if c in dash_sub.columns]
dash_sub = dash_sub[existing]
print(dash_sub.to_string(index=False))
