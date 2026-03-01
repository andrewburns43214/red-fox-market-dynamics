import pandas as pd

s = pd.read_csv("data/snapshots.csv", dtype=str)

mask = s["sport"].str.lower() == "ncaab"
mask &= s["game"].str.contains("alabama", case=False, na=False)
mask &= s["game"].str.contains("miss", case=False, na=False)

print(s.loc[mask, ["game","market","side"]].drop_duplicates())
print("\nTOTAL MATCHES:", mask.sum())
