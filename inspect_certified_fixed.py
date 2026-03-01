import pandas as pd

state = pd.read_csv("data/row_state.csv", keep_default_na=False)

cert = state[state["strong_certified"].astype(str)=="1"]

print("Certified STRONG rows:", len(cert))
print(cert[["sport","game_id","market","side","last_score","last_net_edge"]]
      .head(10).to_string(index=False))
