import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

def team_of(x):
    s = str(x or "").strip()
    if s == "" or s.lower()=="nan": return ""
    # strip spread number if present
    # "BOS Celtics -6.5" -> "BOS Celtics"
    parts = s.rsplit(" ", 1)
    if len(parts)==2 and (parts[1].startswith("+") or parts[1].startswith("-")):
        try:
            float(parts[1])
            return parts[0].strip()
        except: 
            return s
    return s

d["SP_team"] = d["SPREAD_favored"].apply(team_of)
d["ML_team"] = d["MONEYLINE_favored"].apply(team_of)

# Only count as "conflict" when BOTH markets are meaningful (>=60)
def fnum(v):
    try: return float(str(v).strip())
    except: return None

sp_ok = d["SPREAD_model_score"].apply(fnum).fillna(0) >= 60
ml_ok = d["MONEYLINE_model_score"].apply(fnum).fillna(0) >= 60

conf = (d["SP_team"]!="") & (d["ML_team"]!="") & (d["SP_team"]!=d["ML_team"]) & sp_ok & ml_ok

print("rows:", len(d))
print("Meaningful SP+ML conflicts (both >=60):", int(conf.sum()))
print("Sample conflicts:")
print(d.loc[conf, ["game_id","game","SPREAD_favored","SPREAD_model_score","MONEYLINE_favored","MONEYLINE_model_score"]].head(10).to_string(index=False))
