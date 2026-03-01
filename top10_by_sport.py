import pandas as pd

d = pd.read_csv("data/dashboard.csv")

# Ensure numeric
d["game_confidence"] = pd.to_numeric(d["game_confidence"], errors="coerce")
d["net_edge"] = pd.to_numeric(d["net_edge"], errors="coerce")

sports = ["nba", "ncaab", "nhl"]

for sp in sports:
    df = d[d["sport"].str.lower() == sp.lower()].copy()
    
    if df.empty:
        print(f"\n--- {sp.upper()} ---")
        print("No rows found.")
        continue
    
    top = df.sort_values("game_confidence", ascending=False).head(10)
    
    print(f"\n==============================")
    print(f"TOP 10 — {sp.upper()}")
    print(f"==============================")
    
    cols = [
        "sport",
        "game_id",
        "market_display",
        "game_confidence",
        "net_edge",
        "game_decision",
        "favored_side"
    ]
    
    print(top[cols].to_string(index=False))

print("\nDone.")
