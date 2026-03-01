import pandas as pd

s = pd.read_csv("data/snapshots.csv", dtype=str)

# show rows with real finals AND simple team side
real = s[(s["final_score_for"].str.strip() != "")]

print(real[["game","side","market"]].head(20))
