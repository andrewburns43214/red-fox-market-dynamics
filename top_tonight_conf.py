import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

# Load dashboard
d = pd.read_csv("data/dashboard.csv")

# Ensure timestamp parsed
d["_game_time"] = pd.to_datetime(d["_game_time"], errors="coerce")

# Filter strictly to tonight (Eastern date from file)
target_date = pd.Timestamp.now(tz="America/New_York").date()
d = d[d["_game_time"].dt.date == target_date]

if d.empty:
    print("No games found for tonight.")
    exit()

def print_top(df, sport_name, n):
    df = df[df["sport"].str.lower() == sport_name.lower()].copy()
    if df.empty:
        print(f"\n==============================")
        print(f"{sport_name.upper()} — No games tonight")
        print("==============================")
        return

    df = df.sort_values("game_confidence", ascending=False).head(n)

    print(f"\n==============================")
    print(f"{sport_name.upper()} — TOP {n} CONFIDENCE (ALL COLUMNS)")
    print("==============================")
    print(df.to_string(index=False))

# Print required views
print_top(d, "nba", 10)
print_top(d, "ncaab", 20)
print_top(d, "nhl", 10)

print("\nDone.")
