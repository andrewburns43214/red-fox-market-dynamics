import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

print("Total rows:", len(d))
print("BET count:", (d["game_decision"].str.upper()=="BET").sum())

print("\nDecision distribution:")
print(d["game_decision"].value_counts())
