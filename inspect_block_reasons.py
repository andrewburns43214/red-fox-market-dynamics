import pandas as pd

state = pd.read_csv("data/row_state.csv", keep_default_na=False)

print("Block reason counts:")
print(state["strong_block_reasons"].value_counts().head(20))

print("\nRows with non-empty block reasons:")
print((state["strong_block_reasons"].astype(str).str.strip()!="").sum())
print("Total rows:", len(state))
