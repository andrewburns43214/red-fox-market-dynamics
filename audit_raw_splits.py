import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 3000)

s = pd.read_csv("data/snapshots.csv")

# Omaha ML
omaha = s[
    (s["game_id"] == 33702858) &
    (s["market"] == "splits") &
    (s["side"].isin(["Omaha","South Dakota"]))
]

# North Texas Total
nt_total = s[
    (s["game"].str.contains("North Texas @ Charlotte")) &
    (s["side"].str.contains("Under|Over"))
]

print("\n=== OMAHA ML SNAPSHOT ===\n")
print(omaha.tail(6).to_string(index=False))

print("\n=== NORTH TEXAS TOTAL SNAPSHOT ===\n")
print(nt_total.tail(6).to_string(index=False))
