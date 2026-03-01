import pandas as pd

df = pd.read_csv("data/snapshots.csv")

# Convert fake test market to real MLB market
df["market"] = "moneyline"

df.to_csv("data/snapshots.csv", index=False)

print("Converted test MLB markets to MONEYLINE")
