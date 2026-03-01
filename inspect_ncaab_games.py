import pandas as pd

d = pd.read_csv("data/dashboard.csv", dtype=str)

ncaab = d[d["sport"].str.lower() == "ncaab"]

print(ncaab[["game","market_display","favored_side","game_confidence","net_edge","game_decision"]]
      .sort_values(["game","market_display"]))
