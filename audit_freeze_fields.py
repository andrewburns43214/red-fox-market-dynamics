import pandas as pd

r = pd.read_csv("data/results_resolved.csv")

print("Columns:")
print(r.columns.tolist())

print("\nNon-null counts for freeze fields:")
for col in ["game_decision", "net_edge", "game_confidence"]:
    if col in r.columns:
        print(col, "non-null:", r[col].notna().sum())
    else:
        print(col, "NOT PRESENT")

print("\nResolved rows (WIN/LOSS only):")
graded = r[r["result"].isin(["WIN","LOSS"])]
print("graded rows:", len(graded))

print("\nSample graded rows:")
print(graded[[
    "sport","game_id","market_display","side",
    "result","game_decision","net_edge","game_confidence"
]].head(10))
