import pandas as pd

d = pd.read_csv("data/dashboard.csv")

bad = d[d["game_confidence"] > d["max_side_score"]]

print("\nBAD ROWS DETAIL\n")
print(bad[[
    "game","market_display","favored_side",
    "model_score","game_confidence","min_side_score","max_side_score"
]].to_string(index=False))
