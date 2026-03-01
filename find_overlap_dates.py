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

snap["local_date"]=snap["dk_start_iso"].apply(local_date)

snap_dates=set(snap["local_date"])
fin_dates=set(fin["ymd"])

print("Snapshot dates:",sorted(snap_dates)[:20])
print("Finals dates:",sorted(fin_dates)[:20])

print("\nINTERSECTION:")
print(sorted(snap_dates & fin_dates))
