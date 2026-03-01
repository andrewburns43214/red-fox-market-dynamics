import pandas as pd
from main import load_snapshots  # uses same loader as report

df = load_snapshots()

print("SPORT COUNTS BEFORE FILTER:")
print(df["sport"].value_counts(dropna=False))

print("\nSAMPLE MLB ROW:")
print(df[df["sport"]=="mlb"].head(3))
