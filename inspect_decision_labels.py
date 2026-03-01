import pandas as pd
d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

for c in ["SPREAD_decision","MONEYLINE_decision","TOTAL_decision"]:
    vals = sorted(set([str(x).strip() for x in d[c].tolist()]))
    print(c, "unique:", vals)
