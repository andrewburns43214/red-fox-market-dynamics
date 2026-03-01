import pandas as pd
from datetime import datetime, timezone

d = pd.read_csv("data/dashboard.csv")
if "dk_start_iso" not in d.columns:
    print("no dk_start_iso in dashboard.csv")
    raise SystemExit()

kick = pd.to_datetime(d["dk_start_iso"], errors="coerce", utc=True)
now = pd.Timestamp.now(tz="UTC")
hrs = (kick - now).dt.total_seconds()/3600.0

print("kick min/max:", kick.min(), kick.max())
print("hours_to_kick min/max:", float(hrs.min()), float(hrs.max()))
print("<=8h:", int((hrs<=8).sum()), " <=1h:", int((hrs<=1).sum()), " <0:", int((hrs<0).sum()))
print("\nblank timing_bucket rows:", int((d.get("timing_bucket","").fillna("").astype(str).str.strip()=="").sum()))
