import pandas as pd

state = pd.read_csv("data/row_state.csv", keep_default_na=False)

print("Certified STRONG rows:")
print(state.loc[state["strong_certified"]==1,
      ["sport","game_id","market","side","last_score","last_net_edge"]]
      .head(20).to_string(index=False))

print("\nTotal certified strong:", int((state["strong_certified"]==1).sum()))
