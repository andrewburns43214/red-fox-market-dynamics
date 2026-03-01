import pandas as pd

s = pd.read_csv("data/snapshots.csv")

print("\n[SNAPSHOT FINAL SCORE CHECK]")

if "final_score_for" in s.columns and "final_score_against" in s.columns:
    filled = s[(s["final_score_for"].astype(str).str.strip()!="") &
               (s["final_score_against"].astype(str).str.strip()!="")]
    print("Rows with finals:", len(filled))
    print("Total snapshot rows:", len(s))
else:
    print("Final score columns missing")
