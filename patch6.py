with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = "    latest = add_market_pair_checks(latest)\n\n    # --- v1.1 STRONG eligibility (Option B: structure + intent) ---"

new = """    latest = add_market_pair_checks(latest)

    # --- Join row_state into latest so _is_strong_eligible has streak/score/peak data ---
    try:
        _rs = pd.read_csv(ROW_STATE_PATH, dtype=str, keep_default_na=False)
        _rs = _rs[[c for c in ["sport","game_id","market","side","strong_streak","last_score","peak_score"] if c in _rs.columns]].copy()
        _rs = _rs.rename(columns={"market": "market_display"})
        _rs["_rs_side_norm"] = _rs.apply(
            lambda r: normalize_side_key(str(r.get("sport","")), str(r.get("market_display","")), str(r.get("side",""))), axis=1
        )
        latest["_rs_side_norm"] = latest.apply(
            lambda r: normalize_side_key(str(r.get("sport","")), str(r.get("market_display","")), str(r.get("side",""))), axis=1
        )
        _rs_dedup = _rs.drop_duplicates(subset=["sport","game_id","market_display","_rs_side_norm"], keep="last")
        _drop_existing = [c for c in ["strong_streak","last_score","peak_score"] if c in latest.columns]
        if _drop_existing:
            latest = latest.drop(columns=_drop_existing)
        latest = latest.merge(
            _rs_dedup[["sport","game_id","market_display","_rs_side_norm","strong_streak","last_score","peak_score"]],
            on=["sport","game_id","market_display","_rs_side_norm"], how="left"
        )
        latest["strong_streak"] = pd.to_numeric(latest["strong_streak"], errors="coerce").fillna(0).astype(int)
        latest["last_score"] = pd.to_numeric(latest["last_score"], errors="coerce").fillna(0.0)
        latest["peak_score"] = pd.to_numeric(latest["peak_score"], errors="coerce").fillna(0.0)
    except Exception as _rse:
        print(f"[strong] row_state join failed: {_rse}")
        latest["strong_streak"] = 0
        latest["last_score"] = 0.0
        latest["peak_score"] = 0.0
    # --- end row_state join ---

    # --- v1.1 STRONG eligibility (Option B: structure + intent) ---"""

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: row_state joined into latest before _is_strong_eligible")
else:
    print("FAILED: anchor not found")
