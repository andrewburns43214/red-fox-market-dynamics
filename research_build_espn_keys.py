import pandas as pd

f = pd.read_csv("data/finals_espn.csv", dtype=str)

keys=[]
for _,r in f.iterrows():
    key=f'{r["sport"]}|{r["ymd"]}|{r["home"]}|{r["away"]}'
    keys.append(key)

print("UNIQUE ESPN MATCH KEYS:",len(set(keys)))
print(list(set(keys))[:10])
