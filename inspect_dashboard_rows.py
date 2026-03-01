import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

d = pd.read_csv("data/dashboard.csv")

print("\n=== SAMPLE ROWS ===")
print(d.head(5))
