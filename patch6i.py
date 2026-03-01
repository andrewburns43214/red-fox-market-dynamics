with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = """            _rs_row = _rs2_map.get(_ekey, {})
            _ss = int(_rs_row.get("strong_streak", 0)) if _rs_row else 0
            _ls = float(_rs_row.get("last_score", 0)) if _rs_row else 0.0
            _ps = float(_rs_row.get("peak_score", 0)) if _rs_row else 0.0"""

new = """            _rs_row = _rs2_map.get(_ekey, None)
            _ss = 0
            _ls = 0.0
            _ps = 0.0
            if _rs_row is not None:
                try: _ss = int(str(_rs_row.get("strong_streak", 0)).strip() or "0")
                except: _ss = 0
                try: _ls = float(str(_rs_row.get("last_score", 0)).strip() or "0")
                except: _ls = 0.0
                try: _ps = float(str(_rs_row.get("peak_score", 0)).strip() or "0")
                except: _ps = 0.0"""

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS")
else:
    print("FAILED: anchor not found")
