import pandas as pd
r = pd.read_csv("data/results_resolved.csv")
print(r[["sport","game_id"]].head())
