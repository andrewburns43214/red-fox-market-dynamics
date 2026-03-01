import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("\nGAME CONFIDENCE DISTRIBUTION\n")
s = pd.to_numeric(d["game_confidence"], errors="coerce")

print("min:", s.min())
print("avg:", s.mean())
print("max:", s.max())

print("\nCOUNT BY RANGE")
bins = [0,40,50,60,68,72,80,100]
labels = ["<40","40-50","50-60","60-68","68-72","72-80","80+"]
print(pd.cut(s,bins,labels=labels).value_counts().sort_index())

print("\nTOP 20 STRONGEST SIGNALS\n")
print(
    d.sort_values("game_confidence",ascending=False)
    [["sport","game","market_display","favored_side","game_confidence","net_edge","timing_bucket"]]
    .head(20)
)

print("\nLOWEST 20 SIGNALS\n")
print(
    d.sort_values("game_confidence",ascending=True)
    [["sport","game","market_display","favored_side","game_confidence","net_edge","timing_bucket"]]
    .head(20)
)
