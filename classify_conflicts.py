import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

def fnum(v):
    try: return float(str(v).strip())
    except: return 0.0

d["sp"] = d["SPREAD_model_score"].apply(fnum)
d["ml"] = d["MONEYLINE_model_score"].apply(fnum)

conf = (d["sp"]>=60) & (d["ml"]>=60) & (d["SPREAD_favored"]!=d["MONEYLINE_favored"])

print("\nConflicts with BOTH >=72:")
print(d.loc[conf & (d["sp"]>=72) & (d["ml"]>=72),
            ["game_id","game","SPREAD_model_score","MONEYLINE_model_score"]])

print("\nConflicts where one >=72 and other <65:")
print(d.loc[conf & (
    ((d["sp"]>=72) & (d["ml"]<65)) |
    ((d["ml"]>=72) & (d["sp"]<65))
),
["game_id","game","SPREAD_model_score","MONEYLINE_model_score"]])
