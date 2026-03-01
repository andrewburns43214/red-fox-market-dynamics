import pandas as pd, os

path = "data/row_state.csv"
if not os.path.exists(path):
    print("row_state.csv missing")
    raise SystemExit

df = pd.read_csv(path, dtype=str)

# find score column candidates
score_cols = [c for c in df.columns if c.lower() in ("model_score","score","last_score","peak_score") or "model_score" in c.lower()]
print("score cols:", score_cols)

# find strong cols
strong_cols = [c for c in df.columns if "strong" in c.lower()]
print("strong cols:", strong_cols)

# pick a score col to analyze
score_col = "model_score" if "model_score" in df.columns else (score_cols[0] if score_cols else None)
if not score_col:
    print("No usable score column in row_state.")
    raise SystemExit

s = pd.to_numeric(df[score_col], errors="coerce")
print("\nrows with numeric score:", s.notna().sum(), "/", len(df))
print("max score:", float(s.max()) if s.notna().any() else None)
print("count >=72:", int((s>=72).sum()))

# show top rows
top = df.loc[s.sort_values(ascending=False).head(20).index].copy()
show = [c for c in ["sport","game_id","market_display","market","side",score_col,"timing_bucket","decision","row_status"] if c in df.columns] + strong_cols
show = list(dict.fromkeys(show))
print("\nTOP 20 by score:")
print(top[show].to_string(index=False))
