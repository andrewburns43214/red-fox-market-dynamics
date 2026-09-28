import pandas as pd
d = pd.read_csv("data/snapshots_mlb_test.csv")
print("\nCOLUMNS:")
for c in d.columns:
    print(c)
print("\nROWS:", len(d))
print("\nHEAD:")
print(d.head())
