import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("Sports present:")
print(d["sport"].value_counts())
