import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("\nRows by sport:")
print(d.groupby("sport")["game_confidence"].count())

print("\nMean score by sport:")
print(d.groupby("sport")["game_confidence"].mean())

print("\nMax score by sport:")
print(d.groupby("sport")["game_confidence"].max())

print("\nBET count by sport:")
print(d[d["game_decision"]=="BET"].groupby("sport")["game_decision"].count())
