import pandas as pd, re

df = pd.read_csv("data/snapshots.csv", keep_default_na=False, dtype=str)
mlb = df[df["sport"].str.upper()=="MLB"].copy()

pat = re.compile(r"\s[+-]\d+(?:\.\d+)?\s*@\s*[+-]?\d+\s*$")

print("MLB ROWS:", len(mlb))
print()

for i,row in mlb.iterrows():
    cl = str(row.get("current_line",""))
    md = str(row.get("market",""))
    # show only lines containing @ (these are the ones being mis-classified)
    if "@" in cl:
        ok = bool(pat.search(cl.upper()))
        print("market=", md, " ok_spread=", ok)
        print("RAW:", repr(cl))
        print()
