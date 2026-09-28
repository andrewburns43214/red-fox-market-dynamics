import pandas as pd

r = pd.read_csv("data/results_resolved.csv")
f = pd.read_csv("data/finals_espn.csv")

# Pick one NBA game from finals
sample_final = f.iloc[0]
print("\nSample final:")
print(sample_final)

# See if that game name exists in results
matches = r[r["game"].str.contains(sample_final["home"].split()[0], na=False)]
print("\nPotential matches in results:")
print(matches.head())
