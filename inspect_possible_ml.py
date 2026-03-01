import pandas as pd

s = pd.read_csv("data/snapshots.csv")

ml = s[
    (s["side"].str.contains("+", regex=False) == False) &
    (s["side"].str.contains("-", regex=False) == False) &
    (~s["side"].str.contains("Over|Under", case=False, na=False))
]

print("Possible ML rows:")
print(ml[["game_id","game","side"]].head(20))
print("\nCount:", len(ml))
