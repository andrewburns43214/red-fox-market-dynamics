import pandas as pd

rs = pd.read_csv("data/row_state.csv")

print("ROW_STATE COLUMNS:")
for c in rs.columns:
    print(" -", c)

print("\nColumns containing 'edge':")
print([c for c in rs.columns if "edge" in c.lower()])
