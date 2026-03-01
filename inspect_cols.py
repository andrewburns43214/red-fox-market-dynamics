import pandas as pd
d = pd.read_csv("data/dashboard.csv")
cols = [c for c in d.columns if any(x in c.lower() for x in ["line","odds","bets","money","market","side"])]
print(cols)
