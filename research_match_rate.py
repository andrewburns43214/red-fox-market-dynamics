import pandas as pd
from datetime import datetime
import pytz

snap = pd.read_csv("data/snapshots.csv", dtype=str)
fin  = pd.read_csv("data/finals_espn.csv", dtype=str)

def local_date(iso):
    if not isinstance(iso,str) or iso=="":
        return ""
    utc=datetime.fromisoformat(iso.replace("Z","+00:00"))
    est=utc.astimezone(pytz.timezone("US/Eastern"))
    return est.strftime("%Y%m%d")

dk=set()
for _,r in snap.iterrows():
    if "@" not in str(r["game"]): continue
    away,home=[x.strip() for x in r["game"].split("@")]
    dk.add(f'{r["sport"]}|{local_date(r["dk_start_iso"])}|{home}|{away}')

espn=set(f'{r["sport"]}|{r["ymd"]}|{r["home"]}|{r["away"]}' for _,r in fin.iterrows())

matches=dk & espn

print("DK games:",len(dk))
print("ESPN finals:",len(espn))
print("MATCHES:",len(matches))
print("MATCH %:",round(len(matches)/max(len(dk),1)*100,2),"%")

print("\nSAMPLE MATCHES:")
print(list(matches)[:10])
