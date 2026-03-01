with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = "    # Join strong_eligible onto game_view via favored_side canonical key\n    _miss_counter = [0]  # mutable counter avoids global"

new = """    # Rebuild elig_map here using correct latest (has _score_num/game_confidence + row_state join)
    elig_map = {}
    _elig_join_misses = 0
    try:
        _rs2 = pd.read_csv("data/row_state.csv", dtype=str, keep_default_na=False)
        if "market_display" in _rs2.columns:
            _rs2 = _rs2.drop(columns=["market_display"])
        _rs2 = _rs2.rename(columns={"market": "market_display"})
        _rs2["_rs_side_norm"] = _rs2.apply(
            lambda r: normalize_side_key(str(r.get("sport","")), str(r.get("market_display","")), str(r.get("side",""))), axis=1
        )
        _rs2["strong_streak"] = pd.to_numeric(_rs2["strong_streak"], errors="coerce").fillna(0).astype(int)
        _rs2["last_score"] = pd.to_numeric(_rs2["last_score"], errors="coerce").fillna(0.0)
        _rs2["peak_score"] = pd.to_numeric(_rs2["peak_score"], errors="coerce").fillna(0.0)
        _rs2_dedup = _rs2.drop_duplicates(subset=["sport","game_id","market_display","_rs_side_norm"], keep="last")
        _rs2_map = {}
        for _, _rr in _rs2_dedup.iterrows():
            _rk = (str(_rr["sport"]), str(_rr["game_id"]), str(_rr["market_display"]), str(_rr["_rs_side_norm"]))
            _rs2_map[_rk] = _rr
        for _, _er in latest.iterrows():
            _esp = str(_er.get("sport","")).strip()
            _egid = str(_er.get("game_id","")).strip()
            _emkt = str(_er.get("market_display","")).strip()
            _eside = str(_er.get("side","")).strip()
            _esnorm = normalize_side_key(_esp, _emkt, _eside)
            _ekey = (_esp, _egid, _emkt, _esnorm)
            _score = 0.0
            try: _score = float(_er.get("_score_num", _er.get("game_confidence", 0)))
            except: pass
            _rs_row = _rs2_map.get(_ekey, {})
            _ss = int(_rs_row.get("strong_streak", 0)) if _rs_row else 0
            _ls = float(_rs_row.get("last_score", 0)) if _rs_row else 0.0
            _ps = float(_rs_row.get("peak_score", 0)) if _rs_row else 0.0
            _tb = str(_er.get("timing_bucket","")).strip().upper()
            _mr = str(_er.get("market_read","")).strip()
            _pc = str(_er.get("market_pair_check","")).strip()
            _sport_u = _esp.upper()
            # Gate checks
            if _score < 72: continue
            if _tb == "LATE": continue
            if _mr == "Public Drift": continue
            if _pc != "": continue
            if _sport_u == "NCAAB" and _tb == "EARLY": continue
            if _sport_u == "NCAAF" and _tb == "EARLY": continue
            _min_streak = 3 if _sport_u == "NCAAB" else 2
            if _ss < _min_streak: continue
            _delta = 2.0 if _sport_u == "NCAAB" else 3.0
            if _ls < (_ps - _delta): continue
            elig_map[_ekey] = True
    except Exception as _ee:
        print(f"[strong] elig_map rebuild error: {_ee}")
    print(f"[strong] elig_map entries: {len(elig_map)}")

    # Join strong_eligible onto game_view via favored_side canonical key
    _miss_counter = [0]  # mutable counter avoids global"""

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS")
else:
    print("FAILED: anchor not found")
