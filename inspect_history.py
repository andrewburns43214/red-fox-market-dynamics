import pandas as pd

try:
    h = pd.read_csv("data/final_scores_history.csv")
    print("final_scores_history rows:", len(h))
    print("\nColumns:")
    for c in h.columns:
        print("-", c)
    print("\nHead:")
    print(h.head(10))
except Exception as e:
    print("ERROR:", e)
