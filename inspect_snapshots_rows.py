import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

d = pd.read_csv("data/snapshots.csv")

print("\n=== SNAPSHOT SAMPLE ===")
print(d.head(5))
