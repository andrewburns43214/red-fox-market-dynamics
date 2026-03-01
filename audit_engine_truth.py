import pandas as pd

snap = pd.read_csv("data/snapshots.csv", keep_default_na=False)
print("snap rows:", len(snap))

# Grab side-level evaluation before wide reshape
# This assumes latest rows are what metrics consumed.
# So inspect row_state to see actual last_score values.

state = pd.read_csv("data/row_state.csv", keep_default_na=False)
print("row_state sample:")
print(state[["sport","game_id","market","side","last_score","last_net_edge"]].head(10))
print("\nScore range:", state["last_score"].astype(float).min(), state["last_score"].astype(float).max())
