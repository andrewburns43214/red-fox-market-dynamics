import pandas as pd

d = pd.read_csv("data/dashboard.csv")

d["calc_edge"] = d["max_side_score"] - d["min_side_score"]
d["diff"] = d["calc_edge"] - d["net_edge"]

bad = d[d["diff"].abs() > 1e-9]

print("Mismatches with tolerance:", len(bad))

if len(bad) > 0:
    print(bad[["sport","game","market_display",
               "min_side_score","max_side_score",
               "net_edge","calc_edge","diff"]].head())
    
print("\nMax absolute diff:", d["diff"].abs().max())
