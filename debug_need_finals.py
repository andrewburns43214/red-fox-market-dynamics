import pandas as pd

df = pd.read_csv("data/snapshots.csv", dtype=str)

df["final_score_for"] = df["final_score_for"].fillna("").astype(str).str.strip()
df["final_score_against"] = df["final_score_against"].fillna("").astype(str).str.strip()

need = df[(df["final_score_for"] == "") | (df["final_score_against"] == "")]

print("TOTAL rows needing finals:", len(need))
print("Unique games needing finals:", need["game_id"].nunique())
print("\nBy sport:")
print(need.groupby("sport")["game_id"].nunique())
