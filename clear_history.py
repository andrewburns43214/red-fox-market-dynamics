import pandas as pd
from pathlib import Path

p = Path("data/final_scores_history.csv")
pd.DataFrame(columns=["game_id","team1","team1_score","team2","team2_score","resolved_at_utc"]).to_csv(p, index=False)

print("final_scores_history.csv cleared (epoch reset clean).")
