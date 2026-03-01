import pandas as pd

d = pd.read_csv("data/row_state.csv", keep_default_na=False, dtype=str)

# build identity key
d["_id"] = (
    d["sport"] + "|" +
    d["game_id"] + "|" +
    d["market"] + "|" +
    d["side"]
)

print("\nTOTAL ROWS:", len(d))
print("UNIQUE IDS:", d["_id"].nunique())

dup = d[d.duplicated("_id", keep=False)]

print("\nDUPLICATE ROWS:", len(dup))
if len(dup) > 0:
    print("\nSAMPLE DUPLICATES:")
    print(dup[["_id","last_score","peak_score","ts"]].head(20).to_string(index=False))

# detect resets (peak < last impossible)
bad = d[pd.to_numeric(d["peak_score"], errors="coerce") <
        pd.to_numeric(d["last_score"], errors="coerce")]

print("\nPEAK < LAST (should be 0):", len(bad))
