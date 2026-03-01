import pandas as pd
d = pd.read_csv("data/metrics_feed.csv", keep_default_na=False, dtype=str)

print("ROWS:", len(d))
print("\nCOLUMNS:")
for c in d.columns:
    print(c)

print("\nNONBLANK COUNTS:")
for c in d.columns:
    nonblank = (d[c].astype(str).str.strip()!="").sum()
    if nonblank>0:
        print(f"{c}: {nonblank}")

print("\nSAMPLE ROW:")
print(d.head(1).to_string(index=False))
