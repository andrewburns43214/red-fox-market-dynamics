import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print(">=72:", (d["game_confidence"] >= 72).sum())
print(">=68:", (d["game_confidence"] >= 68).sum())
print(">=65:", (d["game_confidence"] >= 65).sum())
print(">=60:", (d["game_confidence"] >= 60).sum())
