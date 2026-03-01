import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)
print("rows:", len(d))
print("cols:", list(d.columns))
print("has market_display:", "market_display" in d.columns)

if "game_id" in d.columns and "market_display" in d.columns:
    print("rows per game_id sample:")
    print(
        d.groupby("game_id")["market_display"]
         .nunique()
         .sort_values(ascending=False)
         .head(10)
    )
