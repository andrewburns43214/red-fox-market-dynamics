import pandas as pd

rs = pd.read_csv("data/row_state.csv", keep_default_na=False, dtype=str)
lg = pd.read_csv("data/signal_ledger.csv", keep_default_na=False, dtype=str)

print("\n[ROW_STATE]")
print("rows:", len(rs), "cols:", len(rs.columns))
for c in ["sport","game_id","market_display","side_key","last_score","peak_score","last_decision"]:
    print(c, "present:", c in rs.columns)

print("\n[SIGNAL_LEDGER]")
print("rows:", len(lg), "cols:", len(lg.columns))
for c in ["ts","sport","game_id","market_display","side_key","event_type","score","logic_version"]:
    print(c, "present:", c in lg.columns)

# show last 10 ledger rows
print("\nLAST 10 LEDGER:")
print(lg.tail(10).to_string(index=False))
