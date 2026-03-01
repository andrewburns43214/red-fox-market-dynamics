import pandas as pd

snap = pd.read_csv("data/snapshots.csv")

if "dk_start_iso" in snap.columns:
    snap["dk_start_iso"] = pd.to_datetime(snap["dk_start_iso"], errors="coerce")

    today = pd.Timestamp.now(tz="UTC").normalize()

    today_games = snap[snap["dk_start_iso"].dt.date == today.date()]

    print("Today's games count:", len(today_games["game_id"].unique()))
    print(today_games[["sport","game_id"]].drop_duplicates().tail(20))
else:
    print("dk_start_iso not present in snapshots.")
