import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False, dtype=str)
mlb = d[d["sport"].str.upper()=="MLB"].copy()

sus = mlb[mlb["market_display"].str.contains("SPREAD", na=False)]
sus = sus[sus["market_read"].str.contains("Contradiction|Reverse|Pressure|Stealth|Freeze", na=False)]

print("MLB SPREAD PRESSURE ROWS:", len(sus))
print(sus[["game","market_display","market_read","market_why"]].head(40).to_string(index=False))
