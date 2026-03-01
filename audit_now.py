import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

def pct(x, denom):
    return 0 if denom==0 else round(100*x/denom, 2)

score = pd.to_numeric(d["game_confidence"], errors="coerce").dropna()

print("ALL rows:", len(d))
print("%>=60:", pct((score>=60).sum(), len(score)))
print("%>=70:", pct((score>=70).sum(), len(score)))
print("72-75 count:", ((score>=72)&(score<=75)).sum())

sp = d[d["market_display"].astype(str).str.upper()=="SPREAD"]
sp_score = pd.to_numeric(sp["game_confidence"], errors="coerce").dropna()

print("\nSPREAD rows:", len(sp))
print("%>=60:", pct((sp_score>=60).sum(), len(sp_score)))
print("%>=70:", pct((sp_score>=70).sum(), len(sp_score)))
print("72-75 count:", ((sp_score>=72)&(sp_score<=75)).sum())
