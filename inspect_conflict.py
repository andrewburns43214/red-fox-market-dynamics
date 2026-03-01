import pandas as pd
d=pd.read_csv("data/dashboard.csv")

bad=d[(d.game_confidence>d.max_side_score)]
print(bad[["game","sport_label","game_confidence","max_side_score","min_side_score"]])
