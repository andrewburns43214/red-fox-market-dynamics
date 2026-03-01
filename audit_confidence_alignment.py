import pandas as pd
import main

# Build side-level latest in memory
latest = main.build_latest_only_for_audit()  # you may need to expose a helper if not already

dash = pd.read_csv("data/dashboard.csv")

# Get favored side model_score from side-level
fav = latest.loc[
    latest["side"] == latest["favored_side"]
]

# Aggregate side-level to get max per market
side_agg = (
    latest
    .groupby(["sport","game_id","market_display"])
    .apply(lambda g: g.loc[g["model_score"].idxmax()])
    .reset_index(drop=True)
)

merged = dash.merge(
    side_agg[["sport","game_id","market_display","model_score"]],
    on=["sport","game_id","market_display"],
    how="left"
)

merged["diff"] = merged["model_score"] - merged["game_confidence"]

print("Mismatches:", (merged["diff"].abs() > 1e-9).sum())
print(merged[merged["diff"].abs() > 1e-9])
