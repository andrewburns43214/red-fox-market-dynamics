import pandas as pd
s = pd.read_csv("data/snapshots.csv")
print("ROWS:", len(s))
print("\nSPORT COUNTS:")
print(s["sport"].value_counts())
print("\nSample games:")
print(s["game"].head(20).to_string(index=False))
