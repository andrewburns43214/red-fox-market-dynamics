import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("ROWS:",len(d))

print("\nGAME DECISION COUNTS:")
print(d["game_decision"].value_counts())

print("\nCONFIDENCE SUMMARY:")
print(d["game_confidence"].describe())

print("\nNET EDGE SUMMARY:")
print(d["net_edge"].describe())

print("\nSCORE RELATIONSHIP CHECK:")
print(d[["min_side_score","max_side_score","game_confidence"]].head(20))

print("\nILLOGICAL STATES (confidence outside bounds):")
bad = d[(d.game_confidence < d.min_side_score) | (d.game_confidence > d.max_side_score)]
print(len(bad))
print(bad[["game","min_side_score","max_side_score","game_confidence"]])

print("\nHIGH CONF LOW EDGE (danger):")
print(d[(d.game_confidence>=68)&(d.net_edge<5)][["game","game_confidence","net_edge"]])

print("\nLOW CONF HIGH EDGE (logic miss):")
print(d[(d.game_confidence<60)&(d.net_edge>=12)][["game","game_confidence","net_edge"]])

print("\nPOTENTIAL CONFLICT GAMES:")
print(d[(d.max_side_score - d.min_side_score)<4][["game","min_side_score","max_side_score","game_decision"]])
