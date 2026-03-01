import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("ROWS:",len(d))
print("\nCOLUMNS:")
print(d.columns.tolist())

print("\nSCORE SUMMARY:")
print(d["model_score"].describe())

print("\nBUCKET COUNTS:")
print(d["decision"].value_counts())

print("\nNULL CHECK:")
print(d.isna().sum().sort_values(ascending=False).head(15))

print("\nEXTREME SCORES:")
print(d.sort_values("model_score",ascending=False).head(10)[["game","market","side","model_score","decision","net_edge"]])
print(d.sort_values("model_score",ascending=True).head(10)[["game","market","side","model_score","decision","net_edge"]])
