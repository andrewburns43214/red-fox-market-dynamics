import pandas as pd

dash = pd.read_csv("data/dashboard.csv", keep_default_na=False)
state = pd.read_csv("data/row_state.csv", keep_default_na=False)

live_ids = dash["game_id"].astype(str).unique().tolist()

eng_live = state[state["game_id"].astype(str).isin(live_ids)].copy()
eng_live = eng_live[["sport","game_id","market","side","last_score","last_net_edge"]]
eng_live = eng_live.sort_values(["game_id","market"])

print("\n=== ENGINE (for live dashboard games) ===")
print(eng_live.head(20).to_string(index=False))

print("\n=== DASHBOARD (live) ===")
print(dash.head(10).to_string(index=False))
