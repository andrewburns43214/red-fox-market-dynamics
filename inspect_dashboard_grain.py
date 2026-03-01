import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("\nTOTAL ROWS:", len(d))

# ---- Test A: Game+Market Grain ----
key_A = ["sport","game_id","market_display"]
dupes_A = d.duplicated(subset=key_A).sum()
unique_A = d[key_A].drop_duplicates().shape[0]

print("\nTEST A — (sport, game_id, market_display)")
print("Unique combos:", unique_A)
print("Duplicate rows under A grain:", dupes_A)

# ---- Test B: Side-Level Grain ----
if "side" in d.columns:
    key_B = ["sport","game_id","market_display","side"]
    dupes_B = d.duplicated(subset=key_B).sum()
    unique_B = d[key_B].drop_duplicates().shape[0]

    print("\nTEST B — (sport, game_id, market_display, side)")
    print("Unique combos:", unique_B)
    print("Duplicate rows under B grain:", dupes_B)
else:
    print("\nNo 'side' column present — cannot be side-level grain.")

