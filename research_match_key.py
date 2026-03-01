import pandas as pd
from datetime import datetime
import pytz

snap = pd.read_csv("data/snapshots.csv", dtype=str)
fin  = pd.read_csv("data/finals_espn.csv", dtype=str)

def ymd_from_iso(iso):
    if not isinstance(iso,str) or not iso:
        return ""
    utc = datetime.fromisoformat(iso.replace("Z","+00:00"))
    est = utc.astimezone(pytz.timezone("US/Eastern"))
    return est.strftime("%Y%m%d")

# DK unique game keys
dk=set()
for _,r in snap.drop_duplicates(["sport","game","dk_start_iso"]).iterrows():
    g=str(r.get("game",""))
    if "@" not in g:
        continue
    away,home=[x.strip() for x in g.split("@",1)]
    ymd=ymd_from_iso(str(r.get("dk_start_iso","")))
    if not ymd:
        continue
    dk.add((str(r["sport"]), ymd, home, away))

# ESPN finals keys
espn=set()
for _,r in fin.iterrows():
    espn.add((str(r["sport"]), str(r["ymd"]), str(r["home"]).strip(), str(r["away"]).strip()))

matches = dk & espn

print("DK unique games:", len(dk))
print("ESPN finals:", len(espn))
print("MATCHES:", len(matches))
if dk:
    print("match %:", round(len(matches)/len(dk)*100,2), "%")

print("\nSAMPLE DK keys:")
print(list(dk)[:10])
print("\nSAMPLE MATCHES:")
print(list(matches)[:10])
