with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = """            # --- v1.1 STRONG streak (instrumentation only) ---
            # strong_streak = consecutive runs where score >= 72
            prev = state_map.get(k, {})
            prev_streak = 0
            try:
                prev_streak = int(str(prev.get("strong_streak","0")).strip() or "0")
            except Exception:
                prev_streak = 0
            try:
                strong_now = (str(bucket).strip().upper() == "STRONG_BET")
            except Exception:
                strong_now = False
            # --- v1.1 FIX: streak must only advance on NEW snapshot tick (idempotent on repeated report runs)
            cur_tick = _metrics_blank(r.get("ts") or r.get("snapshot_ts") or r.get("timestamp"))
            prev_tick = _metrics_blank(prev.get("last_seen_tick") or prev.get("last_tick") or prev.get("last_ts"))
            is_new_tick = (cur_tick != "" and cur_tick != prev_tick)

            if not is_new_tick:
                strong_streak = str(prev_streak if strong_now else 0)
            else:
                strong_streak = str((prev_streak + 1) if strong_now else 0)
            # --- end v1.1 FIX ---
            # --- end v1.1 ---"""

new = """            # --- v1.1 STRONG precheck (true eligibility gate for streak) ---
            # strong_precheck_now = structural gates only (no persistence/stability -- those are circular)
            # Gates: score>=72, not LATE, not PUBLIC DRIFT, no cross-market contradiction, sport early blocks
            prev = state_map.get(k, {})
            prev_streak = 0
            try:
                prev_streak = int(str(prev.get("strong_streak","0")).strip() or "0")
            except Exception:
                prev_streak = 0
            try:
                _tb = str(r.get("timing_bucket","")).strip().upper()
                _mr = str(r.get("market_read","")).strip()
                _pc = str(r.get("market_pair_check","")).strip()
                _sp = str(r.get("sport","")).strip().upper()
                _score_ok = (score >= 72.0)
                _late_ok = (_tb != "LATE")
                _drift_ok = (_mr != "Public Drift")
                _xmkt_ok = (_pc == "")
                # Sport-specific early blocks
                _early_ok = True
                if _sp == "NCAAB" and _tb == "EARLY":
                    _early_ok = False  # NCAAB_EARLY_STRONG_BLOCK
                if _sp == "NCAAF" and _tb == "EARLY":
                    _early_ok = False  # NCAAF_EARLY_INSTANT_STRONG_BLOCK
                strong_precheck_now = (
                    _score_ok and _late_ok and _drift_ok and _xmkt_ok and _early_ok
                )
            except Exception:
                strong_precheck_now = False
            # Streak uses precheck only (Grace-1: one missed tick allowed without reset)
            cur_tick = _metrics_blank(r.get("ts") or r.get("snapshot_ts") or r.get("timestamp"))
            prev_tick = _metrics_blank(prev.get("last_seen_tick") or prev.get("last_tick") or prev.get("last_ts"))
            is_new_tick = (cur_tick != "" and cur_tick != prev_tick)
            prev_miss = 0
            try:
                prev_miss = int(str(prev.get("strong_miss_streak","0")).strip() or "0")
            except Exception:
                prev_miss = 0
            if not is_new_tick:
                # Same tick -- idempotent, do not advance
                strong_streak = str(prev_streak)
                strong_miss_streak = str(prev_miss)
            else:
                if strong_precheck_now:
                    strong_streak = str(prev_streak + 1)
                    strong_miss_streak = "0"
                else:
                    # Grace-1: one miss allowed before streak resets
                    if prev_miss >= 1:
                        strong_streak = "0"
                        strong_miss_streak = "0"
                    else:
                        strong_streak = str(prev_streak)
                        strong_miss_streak = str(prev_miss + 1)
            # --- end v1.1 STRONG precheck ---"""

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: Patch 3 strong_streak gate replaced")
else:
    print("FAILED: anchor not found -- file unchanged")
