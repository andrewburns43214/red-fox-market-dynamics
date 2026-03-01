import pandas as pd
from pprint import pprint

df = pd.read_csv("data/dashboard.csv")

print("\n--- DASHBOARD COLUMNS ---")
print(list(df.columns))

print("\n--- SAMPLE ROW ---")
pprint(df.head(1).to_dict("records")[0])
