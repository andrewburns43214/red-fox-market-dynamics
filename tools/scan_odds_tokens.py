import pandas as pd, re
s = pd.read_csv("data/snapshots.csv", dtype=str, keep_default_na=False)
cl = s["current_line"].astype(str)
pat = cl.str.contains(r"@\s*[+-]\d+", regex=True)
print("Rows with @ +/-digits:", int(pat.sum()))
print("\nSAMPLE:")
print(s.loc[pat, ["timestamp","sport","market","game","side","current_line"]].tail(25).to_string(index=False))
