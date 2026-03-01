import pandas as pd

d = pd.read_csv("data/dashboard.csv")

bad = d[d["max_side_score"] < d["game_confidence"]].copy()

print("BAD ROWS:", len(bad))
if len(bad):
    cols = ["game","sport_label","market_display","model_score","game_confidence","min_side_score","max_side_score","net_edge"]
    for c in cols:
        if c not in bad.columns:
            print("MISSING COL:", c)
    cols = [c for c in cols if c in bad.columns]
    print(bad[cols].to_string(index=False))

    if "model_score" in bad.columns:
        print("\nDoes game_confidence == model_score on bad rows?")
        print((bad["game_confidence"] - bad["model_score"]).describe())
        print("\nUnique deltas:", sorted(set((bad["game_confidence"] - bad["model_score"]).round(4).tolist())))
