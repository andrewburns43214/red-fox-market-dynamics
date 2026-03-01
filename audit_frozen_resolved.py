import pandas as pd

r = pd.read_csv("data/results_resolved.csv")

frozen_resolved = r[
    (r["result"].isin(["WIN","LOSS"])) &
    (r["game_decision"].notna())
]

print("Frozen + resolved rows:", len(frozen_resolved))
print(frozen_resolved[[
    "sport","game_id","market_display","side",
    "result","game_decision","net_edge","game_confidence"
]].head())
