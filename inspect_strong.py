import pandas as pd
d = pd.read_csv("data/dashboard.csv")
print(d[["sport","game","market_display","model_score","strong_eligible","strong_block_reason"]])
