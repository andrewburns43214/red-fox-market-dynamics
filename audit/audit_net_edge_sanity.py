import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

for c in ["SPREAD_model_score","MONEYLINE_model_score","TOTAL_model_score","net_edge"]:
    if c not in d.columns:
        print("missing", c); raise SystemExit(1)

S = d[["SPREAD_model_score","MONEYLINE_model_score","TOTAL_model_score"]].apply(
    lambda col: pd.to_numeric(col, errors="coerce")
)

calc = (S.max(axis=1) - S.min(axis=1)).round(1)
net  = pd.to_numeric(d["net_edge"], errors="coerce").round(1)

bad = d[ (calc.notna()) & (net.notna()) & (calc != net) ]
print("\n[NET_EDGE SANITY]")
print("nonblank net_edge:", net.notna().sum(), "/", len(d))
print("mismatches:", len(bad))

if len(bad):
    out = bad[["game","SPREAD_model_score","MONEYLINE_model_score","TOTAL_model_score","net_edge"]].head(20)
    print(out.to_string(index=False))
