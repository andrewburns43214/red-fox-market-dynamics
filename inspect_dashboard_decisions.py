import pandas as pd

dash = pd.read_csv("data/dashboard.csv", keep_default_na=False)

print(dash.loc[dash["game_id"]=="33694749",
    ["SPREAD_model_score","SPREAD_decision",
     "MONEYLINE_model_score","MONEYLINE_decision"]])
