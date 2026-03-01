import pandas as pd

snap = pd.read_csv("data/snapshots.csv", dtype=str)

print("Columns:", snap.columns.tolist())

print("\nRows with final scores populated:")
print(snap[(snap["final_score_for"] != "") & (snap["final_score_against"] != "")].head(10))

print("\nCount rows with final scores:",
      ((snap["final_score_for"] != "") & (snap["final_score_against"] != "")).sum())
