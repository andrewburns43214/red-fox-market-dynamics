import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

d = pd.read_csv("data/dashboard.csv")

# Parse timestamp
d["_game_time"] = pd.to_datetime(d["_game_time"], errors="coerce")

# Filter strictly to 2/25 Eastern
target_date = pd.Timestamp("2026-02-25").date()
d = d[d["_game_time"].dt.date == target_date]

# Sort within each sport by confidence
sports = ["nba", "ncaab", "nhl"]

for sp in sports:
    df = d[d["sport"].str.lower() == sp.lower()].copy()
    
    if df.empty:
        print(f"\n==============================")
        print(f"{sp.upper()} — No games on 2/25")
        print("==============================")
        continue
    
    df = df.sort_values("game_confidence", ascending=False)
    
    print(f"\n==============================")
    print(f"{sp.upper()} — 2/25 ONLY (ALL COLUMNS)")
    print("==============================")
    print(df.to_string(index=False))

print("\nDone.")
