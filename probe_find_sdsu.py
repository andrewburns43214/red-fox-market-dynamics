import pandas as pd

sn = pd.read_csv("data/snapshots.csv", dtype=str)
sn["timestamp_dt"] = pd.to_datetime(sn["timestamp"], errors="coerce", utc=True)

cut = pd.Timestamp.utcnow() - pd.Timedelta(hours=48)
sn = sn[sn["timestamp_dt"] >= cut].copy()

# Search across common text fields that might contain teams
cols = [c for c in ["game","side","current_line","team1","team2"] if c in sn.columns]
if not cols:
    print("[sdsu] no searchable text cols found. columns:", list(sn.columns))
    raise SystemExit(0)

hay = sn[cols].fillna("").agg(" | ".join, axis=1).str.lower()

hits = sn[hay.str.contains("san diego", na=False) | hay.str.contains("sdsu", na=False)].copy()

print("[sdsu] hits:", len(hits))
if len(hits):
    g = (hits.groupby(["sport","game_id"])
              .agg(rows=("game_id","size"),
                   last_ts=("timestamp_dt","max"))
              .reset_index()
              .sort_values("last_ts", ascending=False))
    print("\n[sdsu] candidate game_ids:")
    print(g.head(20).to_string(index=False))

    # show 10 sample rows from the most recent matching game_id
    top_gid = g.iloc[0]["game_id"]
    sample = hits[hits["game_id"]==top_gid].head(10)[["sport","game_id","timestamp","market_display","side"]].copy()
    print("\n[sdsu] sample rows for most recent candidate:")
    print(sample.to_string(index=False))
