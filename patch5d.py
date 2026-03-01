with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = """                k = "|".join([
                    _metrics_blank(r.get("sport")),
                    _metrics_blank(r.get("game_id")),
                    _metrics_blank(r.get("market")),
                    _metrics_blank(r.get("side")),
                ])
                state_map[k] = r.to_dict()"""

new = """                _ld_sport = _metrics_blank(r.get("sport"))
                _ld_game_id = _metrics_blank(r.get("game_id"))
                _ld_market = _metrics_blank(r.get("market"))
                _ld_side_raw = _metrics_blank(r.get("side"))
                _ld_side_norm = normalize_side_key(_ld_sport, _ld_market, _ld_side_raw)
                k = "|".join([_ld_sport, _ld_game_id, _ld_market, _ld_side_norm])
                state_map[k] = r.to_dict()"""

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: state_map loader uses normalize_side_key()")
else:
    print("FAILED: anchor not found")
