with open("main.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

out = []
i = 0
found = False

while i < len(lines):
    line = lines[i]
    if not found and "    def _get_strong_eligible(r):" in line:
        out.append("    _miss_counter = [0]  # mutable counter avoids global\n")
        out.append("    def _get_strong_eligible(r):\n")
        i += 1
        # Skip old function body until we hit the game_view apply line
        while i < len(lines):
            if "game_view['strong_eligible'] = game_view.apply" in lines[i]:
                break
            i += 1
        # Write corrected function body
        out.append("        _fav = str(r.get('favored_side','')).strip()\n")
        out.append("        _sp = str(r.get('sport','')).strip()\n")
        out.append("        _gid = str(r.get('game_id','')).strip()\n")
        out.append("        _mkt = str(r.get('market_display','')).strip()\n")
        out.append("        _snorm = normalize_side_key(_sp, _mkt, _fav)\n")
        out.append("        _ekey = (_sp, _gid, _mkt, _snorm)\n")
        out.append("        if _ekey not in elig_map:\n")
        out.append("            _miss_counter[0] += 1\n")
        out.append("            return False\n")
        out.append("        return bool(elig_map[_ekey])\n")
        out.append("\n")
        found = True
        continue
    out.append(line)
    # Fix the print line to use _miss_counter
    if found and "print(f'[strong] elig join misses:" in line:
        out[-1] = "    print(f'[strong] elig join misses: {_miss_counter[0]}')\n"
    i += 1

if found:
    with open("main.py", "w", encoding="utf-8") as f:
        f.writelines(out)
    print("SUCCESS: _get_strong_eligible fixed")
else:
    print("FAILED: anchor not found")
