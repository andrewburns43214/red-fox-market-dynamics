with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = "    latest[\"strong_eligible\"] = latest.apply(_is_strong_eligible, axis=1)"
new = """    # DEBUG: trace _is_strong_eligible on high-score rows
    if True:
        for _di, _dr in latest.iterrows():
            _ds = 0.0
            try: _ds = float(_dr.get("game_confidence", _dr.get("model_score", 0)))
            except: pass
            if _ds >= 72:
                _dss = _dr.get("strong_streak", "?")
                _dls = _dr.get("last_score", "?")
                _dps = _dr.get("peak_score", "?")
                _dtb = _dr.get("timing_bucket", "?")
                _dmr = _dr.get("market_read", "?")
                _result = _is_strong_eligible(_dr)
                print(f"[elig debug] {_dr.get('sport')} {_dr.get('game_id')} {_dr.get('market_display')} {_dr.get('side')} | score={_ds} streak={_dss} ls={_dls} ps={_dps} tb={_dtb} mr={_dmr} -> {_result}")
    latest["strong_eligible"] = latest.apply(_is_strong_eligible, axis=1)"""

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS")
else:
    print("FAILED")
