import pandas as pd
d = pd.read_csv("data/dashboard.csv", keep_default_na=False, dtype=str)

d.loc[0, "model_score"] = "999.99"

d.to_csv("data/dashboard.csv", index=False)
print("Injected test score into dashboard")
