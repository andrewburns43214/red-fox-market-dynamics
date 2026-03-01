import pandas as pd
from datetime import datetime
import pytz

d = pd.read_csv("data/snapshots.csv", dtype=str)

def local_date(iso):
    if not isinstance(iso,str) or iso=="":
        return ""
    utc = datetime.fromisoformat(iso.replace("Z","+00:00"))
    est = utc.astimezone(pytz.timezone("US/Eastern"))
    return est.strftime("%Y%m%d")

rows = []
for _,r in d.iterrows():
    if "@" not in str(r["game"]):
        continue

    away,home = [x.strip() for x in r["game"].split("@")]
    date = local_date(r["dk_start_iso"])

    key = f'{r["sport"]}|{date}|{home}|{away}'
    rows.append(key)

print("UNIQUE DK MATCH KEYS:",len(set(rows)))
print(list(set(rows))[:10])
