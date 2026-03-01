import pandas as pd
d = pd.read_csv("data/dashboard.csv", dtype=str)

print(d["game_decision"].value_counts())
