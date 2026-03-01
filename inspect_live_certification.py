import pandas as pd

state = pd.read_csv("data/row_state.csv", keep_default_na=False)
dash = pd.read_csv("data/dashboard.csv", keep_default_na=False)

live_ids = dash["game_id"].astype(str).unique().tolist()

live = state[state["game_id"].astype(str).isin(live_ids)].copy()

def f(x):
    try: return float(str(x))
    except: return 0.0

live["score"] = live["last_score"].apply(f)
live["edge"] = live["last_net_edge"].apply(f)

eligible = live[
    (live["score"] >= 72) &
    (live["strong_block_reasons"].astype(str).str.strip() == "")
]

print("Live rows:", len(live))
print("Eligible STRONG (no block):", len(eligible))
print(eligible[["sport","game_id","market","side","score","edge"]].to_string(index=False))
