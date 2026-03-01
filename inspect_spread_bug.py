import pandas as pd

from pprint import pprint

# We inspect latest inside dashboard builder by recreating partial logic
snap = pd.read_csv("data/snapshots.csv", keep_default_na=False)

# Re-run just enough to inspect side rows for a specific game
# Instead of replicating whole pipeline, inspect dashboard.csv pre-wide pivot

dash = pd.read_csv("data/dashboard.csv", keep_default_na=False)

print("Dashboard SPREAD row for 33682100:")
print(dash[dash["game_id"].astype(str)=="33682100"][["SPREAD_favored","SPREAD_model_score","SPREAD_net_edge"]])
