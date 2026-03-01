import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False, dtype=str)

print("\n[DASHBOARD CONTRACT]")
print("rows:", len(d), "cols:", len(d.columns))
print("cols:", list(d.columns))

req = [
  "sport","game_id","game","sport_label","dk_start_iso",
  "SPREAD_favored","SPREAD_model_score","SPREAD_decision",
  "MONEYLINE_favored","MONEYLINE_model_score","MONEYLINE_decision",
  "TOTAL_favored","TOTAL_model_score","TOTAL_decision",
  "net_edge","timing_bucket"
]
missing = [c for c in req if c not in d.columns]
print("missing:", missing)

# blanks audit
def blank_count(col):
    if col not in d.columns: return None
    return int((d[col].fillna("").astype(str).str.strip()=="").sum())

for c in ["SPREAD_model_score","MONEYLINE_model_score","TOTAL_model_score","net_edge","timing_bucket"]:
    if c in d.columns:
        print(f"blank[{c}]=", blank_count(c))

# quick sample
print("\nSAMPLE:")
print(d.head(5).to_string(index=False))
