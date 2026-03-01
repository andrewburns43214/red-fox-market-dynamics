import pandas as pd
d = pd.read_csv("data/dashboard.csv", dtype=str)

d["max_side_score"] = pd.to_numeric(d["max_side_score"], errors="coerce")
d["net_edge"] = pd.to_numeric(d["net_edge"], errors="coerce")

hi = d[d["max_side_score"] >= 72].copy()
print("dashboard rows with max_side_score>=72:", len(hi))

if len(hi):
    print("\nDecision breakdown among >=72:")
    print(hi["game_decision"].value_counts(dropna=False))

    print("\nNet edge buckets among >=72:")
    bins = [-1,4,8,12,999]
    labels = ["0-4","5-8","9-12","13+"]
    hi["ne_bucket"] = pd.cut(hi["net_edge"].fillna(-1), bins=bins, labels=labels)
    print(hi["ne_bucket"].value_counts(dropna=False))

    cols = [c for c in ["sport","market_display","game","favored_side","max_side_score","net_edge","game_decision"] if c in hi.columns]
    print("\nSample >=72 rows:")
    print(hi[cols].head(25).to_string(index=False))
