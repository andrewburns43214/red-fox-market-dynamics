import pandas as pd

# Load files
d13 = pd.read_csv("data/dashboard_FEB13.csv")
d23 = pd.read_csv("data/dashboard_FEB23.csv")

# --- Convert FEB23 from WIDE to LONG ---
records = []

for _, row in d23.iterrows():
    for market in ["SPREAD","MONEYLINE","TOTAL"]:
        score_col = f"{market}_model_score"
        edge_col  = f"{market}_net_edge"
        if score_col in d23.columns:
            records.append({
                "sport": row["sport"],
                "game_id": row["game_id"],
                "market_display": market,
                "game_confidence_23": row.get(score_col),
                "net_edge_23": row.get(edge_col)
            })

d23_long = pd.DataFrame(records)

# --- Prep FEB13 ---
d13_small = d13[[
    "sport","game_id","market_display",
    "game_confidence","net_edge"
]].copy()

d13_small.rename(columns={
    "game_confidence":"game_confidence_13",
    "net_edge":"net_edge_13"
}, inplace=True)

# --- Merge ---
merged = d13_small.merge(
    d23_long,
    on=["sport","game_id","market_display"],
    how="inner"
)

merged["confidence_diff"] = merged["game_confidence_23"] - merged["game_confidence_13"]
merged["net_edge_diff"] = merged["net_edge_23"] - merged["net_edge_13"]

print("\n===== SUMMARY =====")
print("Rows compared:", len(merged))
print("Avg Feb13:", merged["game_confidence_13"].mean())
print("Avg Feb23:", merged["game_confidence_23"].mean())
print("Max diff:", merged["confidence_diff"].max())
print("Min diff:", merged["confidence_diff"].min())

print("\n===== TOP SCORE CHANGES =====")
print(
    merged.sort_values("confidence_diff", ascending=False)[[
        "sport","game_id","market_display",
        "game_confidence_13",
        "game_confidence_23",
        "confidence_diff",
        "net_edge_13",
        "net_edge_23",
        "net_edge_diff"
    ]].head(15)
)
