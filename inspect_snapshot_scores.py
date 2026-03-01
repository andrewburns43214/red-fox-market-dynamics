import pandas as pd

# this is the pre-aggregation dataset saved by report
d = pd.read_csv("data/snapshots.csv")

# focus problem game
games = [
"Clemson @ Duke",
"UCLA @ Michigan",
"Purdue @ Iowa",
"Texas Tech @ Arizona",
"Gonzaga @ Santa Clara"
]

subset = d[d["game"].isin(games)]

print("\nSNAPSHOT SCORES\n")
cols = [c for c in subset.columns if "score" in c.lower()]
print(cols)

print(subset[["game","market_display"] + cols].head(40).to_string(index=False))
