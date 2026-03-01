import pandas as pd

d = pd.read_csv("data/dashboard.csv")

if "sport_label" not in d.columns:
    print("No sport column found — cannot audit")
    exit()

print("\n================ CONFIDENCE INTEGRITY AUDIT ================\n")

sports = sorted(d["sport_label"].dropna().unique())

total_bad = 0

for s in sports:
    sub = d[d["sport_label"] == s]

    bad = sub[(sub.game_confidence < sub.min_side_score) |
              (sub.game_confidence > sub.max_side_score)]

    print(f"SPORT: {s}")
    print(f"games: {len(sub)}")
    print(f"bad rows: {len(bad)}")

    if len(bad) > 0:
        print("examples:")
        print(bad[["game","game_confidence","min_side_score","max_side_score","net_edge"]].head(5))
        print()

    total_bad += len(bad)

print("------------------------------------------------------------")
print("TOTAL BAD ROWS:", total_bad)
print("============================================================")
