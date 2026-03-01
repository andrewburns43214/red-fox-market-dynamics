import pandas as pd

hist = pd.read_csv("data/final_scores_history.csv", dtype=str)
freeze = pd.read_csv("data/decision_freeze_ledger.csv", dtype=str)

print("\n--- FINAL HISTORY SAMPLE ---")
print(hist[["sport","game_id","game"]].head(10))

print("\n--- FREEZE LEDGER SAMPLE ---")
print(freeze[["sport","game_id","game"]].head(10))
