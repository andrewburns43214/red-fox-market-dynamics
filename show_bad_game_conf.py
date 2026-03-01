import pandas as pd

d = pd.read_csv("data/dashboard.csv")

bad = d[(d.game_confidence < d.min_side_score) | (d.game_confidence > d.max_side_score)]

print("BAD ROW COUNT:",len(bad))
print()

for _,r in bad.iterrows():
    print("GAME:",r["game"])
    print("CONF:",r["game_confidence"])
    print("MIN :",r["min_side_score"])
    print("MAX :",r["max_side_score"])
    print("EDGE:",r["net_edge"])
    print("-"*40)
