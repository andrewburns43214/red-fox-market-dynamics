import pandas as pd

sl = pd.read_csv("data/signal_ledger.csv")

print("Most recent ledger timestamps:")
print(sl.sort_values("ts", ascending=False).head(10)[["ts","game_id","market","side","event"]])
