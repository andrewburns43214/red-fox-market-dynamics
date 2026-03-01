import pandas as pd

d = pd.read_csv("data/dashboard.csv")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

print(d.sort_values(["sport","_game_time","market_display"]))
