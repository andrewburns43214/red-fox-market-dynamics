import pandas as pd

d = pd.read_csv("data/row_state.csv")

print("\nROW_STATE COLUMNS:")
for c in d.columns:
    print(c)

print("\nROW COUNT:", len(d))
