import pandas as pd

dash = pd.read_csv("data/dashboard.csv", keep_default_na=False)
state = pd.read_csv("data/row_state.csv", keep_default_na=False)

dash_ids = set(dash["game_id"].astype(str))
state_ids = set(state["game_id"].astype(str))

print("Dashboard game count:", len(dash_ids))
print("Row_state total game count:", len(state_ids))
print("Overlap count:", len(dash_ids & state_ids))
print("Dashboard missing in row_state:", dash_ids - state_ids)
print("Row_state missing in dashboard:", list(state_ids - dash_ids)[:10])
