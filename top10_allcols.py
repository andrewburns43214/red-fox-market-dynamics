import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.max_colwidth", None)

d = pd.read_csv("data/dashboard.csv")

# Ensure numeric sorting works correctly
d["game_confidence"] = pd.to_numeric(d["game_confidence"], errors="coerce")

sports = ["nba", "ncaab", "nhl"]

for sp in sports:
    df = d[d["sport"].str.lower() == sp.lower()].copy()
    
    if df.empty:
        print(f"\n==============================")
        print(f"{sp.upper()} — No rows found")
        print("==============================")
        continue
    
    top = df.sort_values("game_confidence", ascending=False).head(10)
    
    print(f"\n==============================")
    print(f"TOP 10 — {sp.upper()} (ALL COLUMNS)")
    print("==============================")
    print(top.to_string(index=False))

print("\nDone.")
