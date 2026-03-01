import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

def dec_from_score(x):
    try:
        v = float(x)
    except Exception:
        return "NO BET"
    if v >= 72: return "STRONG"
    if v >= 68: return "BET"
    if v >= 60: return "LEAN"
    return "NO BET"

print("\n[DECISION CONSISTENCY]")
for m in ["SPREAD","MONEYLINE","TOTAL"]:
    sc = f"{m}_model_score"
    dc = f"{m}_decision"
    if sc not in d.columns or dc not in d.columns:
        print(m, "missing cols"); continue
    exp = d[sc].apply(dec_from_score)
    bad = d[exp != d[dc]]
    print(m, "mismatches:", len(bad))
    if len(bad):
        print(bad[[ "game", sc, dc ]].head(20).to_string(index=False))
