import pandas as pd

dash = pd.read_csv("data/dashboard.csv", keep_default_na=False)

strongs = dash[
    (dash["SPREAD_decision"]=="STRONG") |
    (dash["MONEYLINE_decision"]=="STRONG") |
    (dash["TOTAL_decision"]=="STRONG")
]

print("Dashboard STRONG rows:", len(strongs))
print(strongs[[
    "sport","game_id","game",
    "SPREAD_decision","MONEYLINE_decision","TOTAL_decision"
]].to_string(index=False))
