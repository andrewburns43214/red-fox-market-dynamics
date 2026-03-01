import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

strong = d[d["game_decision"].astype(str).str.upper()=="BET"]

print("Total rows:", len(d))
print("BET count:", len(strong))

print("\nBET by timing bucket:")
print(strong["timing_bucket"].value_counts())
