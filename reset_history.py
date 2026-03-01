import pandas as pd
pd.DataFrame(columns=["game_id","team1","team1_score","team2","team2_score","resolved_at_utc"]).to_csv("data/final_scores_history.csv", index=False)
print("final history reset clean.")
