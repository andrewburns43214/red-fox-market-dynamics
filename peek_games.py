import pandas as pd
d=pd.read_csv("data/signal_ledger.csv",nrows=10)
print(d["game"].to_string())
