import pandas as pd

snap = pd.read_csv("data/snapshots.csv", dtype=str, keep_default_na=False)

mask = (
    (snap["final_score_for"].str.strip() != "") &
    (snap["final_score_against"].str.strip() != "")
)

print("Rows with true populated finals:", mask.sum())

print("\nSample of populated rows:")
print(snap[mask].head(10))
