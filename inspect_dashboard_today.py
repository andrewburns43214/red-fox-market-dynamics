import pandas as pd
d = pd.read_csv("data/dashboard.csv")

print("Total dashboard rows:", len(d))

today = pd.Timestamp.now(tz="America/New_York").date()

kick = pd.to_datetime(d["dk_start_iso"], errors="coerce", utc=True).dt.tz_convert("America/New_York")

print("\nGames today NY:")
print(d.loc[kick.dt.date == today, ["sport","game","dk_start_iso"]])
