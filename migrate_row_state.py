import pandas as pd
import shutil
from pathlib import Path

ROW_STATE_PATH = Path("data/row_state.csv")
BACKUP_PATH = Path("data/backups/row_state_premigration.csv")
COLLISION_LOG_PATH = Path("data/row_state_migration_collisions.csv")

# --- Import normalize_side_key from main.py ---
import importlib.util, sys
spec = importlib.util.spec_from_file_location("main", "main.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
normalize_side_key = mod.normalize_side_key
print("[migrate] loaded normalize_side_key from main.py")

# --- Load row_state ---
if not ROW_STATE_PATH.exists():
    print("[migrate] row_state.csv not found -- nothing to migrate")
    raise SystemExit(0)

df = pd.read_csv(ROW_STATE_PATH, keep_default_na=False, dtype=str)
print(f"[migrate] loaded {len(df)} rows from row_state.csv")

# --- Backup before touching anything ---
BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROW_STATE_PATH, BACKUP_PATH)
print(f"[migrate] backup saved to {BACKUP_PATH}")

# --- Compute canonical side_key_norm for every row ---
def compute_key(row):
    return normalize_side_key(
        row.get("sport", ""),
        row.get("market", ""),
        row.get("side", "")
    )

df["side_key_norm"] = df.apply(compute_key, axis=1)

# --- Build new canonical row_key ---
df["_new_key"] = (
    df["sport"].str.strip() + "|" +
    df["game_id"].str.strip() + "|" +
    df["market"].str.strip() + "|" +
    df["side_key_norm"]
)

# --- Detect collisions ---
dupes = df[df.duplicated(subset=["_new_key"], keep=False)].copy()
collision_rows = []

if len(dupes) > 0:
    print(f"[migrate] WARNING: {len(dupes)} rows have canonical key collisions -- resolving")
    for key, group in dupes.groupby("_new_key"):
        group = group.copy()
        def safe_int(x):
            try: return int(str(x).strip() or "0")
            except: return 0
        group["_streak_int"] = group["strong_streak"].apply(safe_int) if "strong_streak" in group.columns else 0
        group["_ts_str"] = group["last_ts"].fillna("") if "last_ts" in group.columns else ""
        winner = group.sort_values(["_streak_int","_ts_str"], ascending=[False,False]).iloc[0]
        losers = group[group.index != winner.name]
        for _, loser in losers.iterrows():
            collision_rows.append({
                "canonical_key": key,
                "kept_old_side": winner.get("side",""),
                "dropped_old_side": loser.get("side",""),
                "kept_strong_streak": winner.get("strong_streak","0"),
                "dropped_strong_streak": loser.get("strong_streak","0"),
                "kept_last_ts": winner.get("last_ts",""),
                "reason": "picked_highest_streak_then_latest_ts"
            })
    df = df.sort_values("_new_key", kind="mergesort").drop_duplicates(subset=["_new_key"], keep="first")
else:
    print("[migrate] no collisions detected")

# --- Write collision log ---
if collision_rows:
    pd.DataFrame(collision_rows).to_csv(COLLISION_LOG_PATH, index=False)
    print(f"[migrate] collision log written: {len(collision_rows)} dropped rows logged")
else:
    print("[migrate] no collision log needed")

# --- Update side to canonical key, drop temp cols ---
df["side"] = df["side_key_norm"]
df = df.drop(columns=["side_key_norm","_new_key","_streak_int","_ts_str"], errors="ignore")

# --- Write upgraded row_state ---
df.to_csv(ROW_STATE_PATH, index=False)
print(f"[migrate] wrote {len(df)} rows to row_state.csv")
print("[migrate] DONE -- row_state is now on canonical keys")
