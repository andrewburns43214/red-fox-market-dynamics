import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("Rows:", len(d))
print("\nColumns:", list(d.columns))

print("\nSample rows:")
print(d.head(10))

print("\nScore distribution:")
print(d["game_confidence"].describe())

print("\nDecision counts:")
print(d["game_decision"].value_counts())
