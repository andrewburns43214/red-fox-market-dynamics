import pandas as pd
d = pd.read_csv("data/dashboard.csv")

print("\nRows where game_confidence == model_score but != max_side_score:\n")
x = d[(d.game_confidence==d.model_score) & (d.game_confidence!=d.max_side_score)]
print(x[["game","market_display","game_confidence","max_side_score","model_score"]])
