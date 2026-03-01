import pandas as pd
import numpy as np

d = pd.read_csv("data/dashboard.csv")
cols = {c.lower(): c for c in d.columns}

def pick(*names):
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None

sport_c = pick("sport")
dec_c = pick("decision", "game_decision")
conf_c = pick("game_confidence", "confidence", "game_conf")
edge_c = pick("net_edge_market", "net_edge")

for c in [conf_c, edge_c]:
    d[c] = pd.to_numeric(d[c], errors="coerce")

print("Decision ranges (overall):")
g = d.groupby(dec_c)[conf_c]
out = pd.DataFrame({
    "count": g.size(),
    "min_conf": g.min(),
    "p25": g.quantile(0.25),
    "median": g.median(),
    "p75": g.quantile(0.75),
    "max_conf": g.max(),
})
print(out.sort_values("min_conf", ascending=False).to_string())

print("\nDecision ranges by sport:")
for sp, s in d.groupby(sport_c):
    g = s.groupby(dec_c)[conf_c]
    out = pd.DataFrame({
        "count": g.size(),
        "min_conf": g.min(),
        "median": g.median(),
        "max_conf": g.max(),
    })
    print(f"\n--- {sp} ---")
    print(out.sort_values("min_conf", ascending=False).to_string())
