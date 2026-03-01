import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

print("\nCHECKING REQUIRED FIELDS\n")
for col in ["sport","game_id","market_display","favored_side","game_confidence"]:
    print(col, "exists=", col in d.columns)

print("\nSAMPLE VALUES\n")
print(d[["sport","game_id","market_display","favored_side","game_confidence"]].head(10))
