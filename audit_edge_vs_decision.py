import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("\nAverage edge by decision:")
print(d.groupby("game_decision")["net_edge"].mean())

print("\nEdge summary by decision:")
print(d.groupby("game_decision")["net_edge"].describe())
