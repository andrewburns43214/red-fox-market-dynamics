import pandas as pd

base = pd.read_csv("data/signals_baseline.csv", dtype=str)
snaps = pd.read_csv("data/snapshots.csv", dtype=str)

base["side"] = base["side"].astype(str)
snaps["side"] = snaps["side"].astype(str)

# Pick a game with finals
test_game = "Mississippi State @ Alabama"

b = base[base["game"] == test_game][["game_id","side"]]
s = snaps[snaps["game"] == test_game][["game_id","side"]]

print("\nBASE SIDES:")
print(b)

print("\nSNAP SIDES:")
print(s)

print("\nRaw repr comparison:")
for x in b["side"].unique():
    print("BASE:", repr(x))
for x in s["side"].unique():
    print("SNAP:", repr(x))
