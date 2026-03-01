import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False, dtype=str)
mlb = d[d["sport"].str.upper()=="MLB"].copy()

sp = mlb[mlb["market_display"].str.contains("SPREAD", na=False)]
ml = mlb[mlb["market_display"].str.contains("MONEYLINE", na=False)]

merged = sp.merge(ml, on="game", suffixes=("_SP","_ML"))

disagree = merged[merged["market_favors_SP"] != merged["market_favors_ML"]]

print("SPREAD vs ML disagreements:", len(disagree))
print(disagree[[
    "game",
    "market_read_SP","market_favors_SP",
    "market_read_ML","market_favors_ML"
]].head(40).to_string(index=False))
