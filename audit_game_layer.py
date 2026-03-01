import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("\n================ GAME AGGREGATION AUDIT ================\n")

bad = []
summary = []

for (game, sport), sub in d.groupby(["game","sport_label"]):

    # best market score
    true_max = sub["max_side_score"].max()

    # stored game confidence (same across rows)
    gc = sub["game_confidence"].iloc[0]

    if abs(gc - true_max) > 0.01:
        bad.append((game, sport, gc, true_max))

    # find which market produced it
    winning_market = sub.loc[sub["max_side_score"].idxmax(),"market_display"]

    summary.append((game, sport, gc, winning_market))

print("Bad games (REAL errors):", len(bad))
for b in bad:
    print("ERROR:", b)

print("\nSample aggregation results:")
for s in summary[:10]:
    print(s)

print("\n========================================================")
