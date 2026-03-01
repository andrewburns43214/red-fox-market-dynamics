import pandas as pd

d = pd.read_csv("data/dashboard.csv")

# recompute
d["calc_edge"] = d["max_side_score"] - d["min_side_score"]

# difference
d["edge_diff"] = d["calc_edge"] - d["net_edge"]

print("Rows where net_edge != max - min:")
bad = d[d["edge_diff"].abs() > 1e-9]

print("Count mismatches:", len(bad))

if len(bad) > 0:
    print(bad[["sport","game","market_display","min_side_score","max_side_score","net_edge","calc_edge"]])

print("\nNet edge distribution:")
print(d["net_edge"].value_counts().sort_index())
