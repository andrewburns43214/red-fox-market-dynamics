import pandas as pd
s = pd.read_csv("data/row_state.csv")

print("\nSample values:")
print(s[["sport","game_id","market","side","last_net_edge"]].head(15))
