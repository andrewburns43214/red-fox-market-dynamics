import pandas as pd
from datetime import datetime, timedelta, timezone

df = pd.read_csv("data/snapshots.csv")

soon = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()

df["dk_start_iso"] = soon

df.to_csv("data/snapshots.csv", index=False)

print("MLB kickoff moved into active window:", soon)
