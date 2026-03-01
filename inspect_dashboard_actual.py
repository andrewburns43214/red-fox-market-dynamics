import pandas as pd
d = pd.read_csv("data/dashboard.csv")
print("\nCOLUMNS:\n")
for c in d.columns:
    print(c)
print("\n\nHEAD:\n")
print(d.head(10))
