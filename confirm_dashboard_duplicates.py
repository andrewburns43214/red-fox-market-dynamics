import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("\nChecking duplicate game+market rows in dashboard:\n")
print(d.groupby(["game","market_display"]).size().sort_values(ascending=False).head(20))
