import pandas as pd
d = pd.read_csv("data/row_state.csv", keep_default_na=False, dtype=str)

print("ROWS:", len(d))
print("\nCOLUMNS:")
for c in d.columns:
    print(c)

print("\nSAMPLE:")
print(d.head(5).to_string(index=False))
