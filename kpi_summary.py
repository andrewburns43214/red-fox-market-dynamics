import pandas as pd

df = pd.read_csv("data/results_resolved.csv", dtype=str)

if df.empty:
    print("No resolved games yet.")
else:
    decided = df[df["result"].isin(["WIN","LOSS","PUSH"])].copy()

    if decided.empty:
        print("No decided games yet.")
    else:
        summary = (
            decided.groupby("game_decision")["result"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )

        summary["total"] = summary.sum(axis=1, numeric_only=True)

        if "WIN" in summary.columns and "LOSS" in summary.columns:
            summary["win_rate_ex_push"] = (
                summary["WIN"] /
                (summary["WIN"] + summary["LOSS"])
            ).round(3)

        print(summary)
