import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

markets = ["SPREAD","MONEYLINE","TOTAL"]

rows = []

for _, r in d.iterrows():
    for m in markets:
        if str(r.get(f"{m}_decision","")).strip() == "STRONG":
            rows.append({
                "sport": r["sport"],
                "game": r["game"],
                "market": m,
                "score": r.get(f"{m}_model_score"),
                "timing_bucket": r.get("timing_bucket"),
                "net_edge": r.get(f"{m}_net_edge")
            })

print("STRONG rows:", len(rows))
for x in rows[:20]:
    print(x)
