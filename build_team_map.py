import pandas as pd

ledger = pd.read_csv("data/signal_ledger.csv", dtype=str)
finals = pd.read_csv("data/finals_espn.csv", dtype=str)

ledger_teams = set()
for g in ledger["game"].dropna():
    if "@" in g:
        a,h=[x.strip() for x in g.split("@")]
        ledger_teams.add(a)
        ledger_teams.add(h)

espn_teams=set(finals["home"])|set(finals["away"])

out=pd.DataFrame({"ledger_name":sorted(ledger_teams)})
out["espn_name"]=""
out.to_csv("data/team_map_template.csv",index=False)

print("ledger teams:",len(ledger_teams))
print("espn teams:",len(espn_teams))
print("created data/team_map_template.csv")
