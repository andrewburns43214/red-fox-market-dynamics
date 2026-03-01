import pandas as pd
d = pd.read_csv("data/snapshots_mlb_test.csv")
print("COLUMNS:", d.columns.tolist())
print("\nFIRST ROW:")
print(d.head(1))
