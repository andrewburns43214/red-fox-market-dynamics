import pandas as pd
from datetime import datetime, timezone

d = pd.read_csv("data/dashboard.csv", keep_default_na=False, dtype=str)

print("\n[CONTRACT BEHAVIOR CHECK]")
if "dk_start_iso" not in d.columns:
    print("missing dk_start_iso"); raise SystemExit(1)

# parse times
def parse_iso(x):
    try:
        return datetime.fromisoformat(x.replace("Z","+00:00"))
    except Exception:
        return None

times = d["dk_start_iso"].apply(parse_iso)
now = datetime.now(timezone.utc)

future = times.apply(lambda t: (t is not None) and (t > now))
print("future games in dashboard:", int(future.sum()), "/", len(d))

# show a few future games
idx = d[future].head(10).index
if len(idx):
    print("\nSAMPLE FUTURE GAMES:")
    print(d.loc[idx, ["sport","game","dk_start_iso","SPREAD_decision","TOTAL_decision","MONEYLINE_decision"]].to_string(index=False))
else:
    print("\nNo future games detected (could be normal depending on schedule window)")
