with open("main.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

out = []
i = 0
found_elig = False
found_decision = False

while i < len(lines):
    line = lines[i]

    # Insertion point 1: after latest["strong_eligible"] line -- build elig_map
    if not found_elig and '    latest["strong_eligible"] = latest.apply(_is_strong_eligible, axis=1)' in line:
        out.append(line)
        out.append("\n")
        out.append("    # --- Patch 5: build elig_map for game_view join ---\n")
        out.append("    # Key: (sport, game_id, market_display, side_key_norm) -> strong_eligible bool\n")
        out.append("    elig_map = {}\n")
        out.append("    _elig_join_misses = 0\n")
        out.append("    try:\n")
        out.append("        for _, _er in latest.iterrows():\n")
        out.append("            _ekey = (\n")
        out.append("                str(_er.get('sport','')).strip(),\n")
        out.append("                str(_er.get('game_id','')).strip(),\n")
        out.append("                str(_er.get('market_display','')).strip(),\n")
        out.append("                normalize_side_key(\n")
        out.append("                    str(_er.get('sport','')).strip(),\n")
        out.append("                    str(_er.get('market_display','')).strip(),\n")
        out.append("                    str(_er.get('side','')).strip()\n")
        out.append("                )\n")
        out.append("            )\n")
        out.append("            elig_map[_ekey] = bool(_er.get('strong_eligible', False))\n")
        out.append("    except Exception as _ee:\n")
        out.append("        print(f'[strong] elig_map build error: {_ee}')\n")
        out.append("    # --- end elig_map ---\n")
        out.append("\n")
        found_elig = True
        i += 1
        continue

    # Replacement point: _game_decision function + game_view apply
    if not found_decision and '    # Game decision: BET requires strong score + meaningful net edge; otherwise LEAN/NO BET' in line:
        # Write replacement block
        out.append("    # Game decision: STRONG_BET requires eligibility + net edge; BET/LEAN/NO BET otherwise\n")
        out.append("    def _game_decision(score, net_edge, strong_eligible=False):\n")
        out.append("        try:\n")
        out.append("            s = float(score)\n")
        out.append("        except Exception:\n")
        out.append("            s = 50.0\n")
        out.append("        try:\n")
        out.append("            ne = float(net_edge)\n")
        out.append("        except Exception:\n")
        out.append("            ne = 0.0\n")
        out.append("        if s >= 72 and ne >= 10 and bool(strong_eligible):\n")
        out.append("            return 'STRONG_BET'\n")
        out.append("        if s >= 72 and ne >= 10:\n")
        out.append("            return 'BET'\n")
        out.append("        if s >= 62:\n")
        out.append("            return 'LEAN'\n")
        out.append("        return 'NO BET'\n")
        out.append("\n")
        out.append("    # Join strong_eligible onto game_view via favored_side canonical key\n")
        out.append("    def _get_strong_eligible(r):\n")
        out.append("        global _elig_join_misses\n")
        out.append("        _fav = str(r.get('favored_side','')).strip()\n")
        out.append("        _sp = str(r.get('sport','')).strip()\n")
        out.append("        _gid = str(r.get('game_id','')).strip()\n")
        out.append("        _mkt = str(r.get('market_display','')).strip()\n")
        out.append("        _snorm = normalize_side_key(_sp, _mkt, _fav)\n")
        out.append("        _ekey = (_sp, _gid, _mkt, _snorm)\n")
        out.append("        if _ekey not in elig_map:\n")
        out.append("            _elig_join_misses += 1\n")
        out.append("            return False\n")
        out.append("        return elig_map[_ekey]\n")
        out.append("\n")
        out.append("    game_view['strong_eligible'] = game_view.apply(_get_strong_eligible, axis=1)\n")
        out.append("\n")
        found_decision = True
        # Skip old _game_decision function and old game_view apply line
        i += 1
        while i < len(lines):
            if 'game_view["game_decision"] = game_view.apply' in lines[i]:
                # Write new apply line
                out.append("    game_view['game_decision'] = game_view.apply(\n")
                out.append("        lambda r: _game_decision(\n")
                out.append("            r.get('game_confidence', 50),\n")
                out.append("            r.get('net_edge', 0),\n")
                out.append("            r.get('strong_eligible', False)\n")
                out.append("        ), axis=1\n")
                out.append("    )\n")
                out.append("    print(f'[strong] elig join misses: {_elig_join_misses}')\n")
                i += 1
                break
            i += 1
        continue

    out.append(line)
    i += 1

if found_elig and found_decision:
    with open("main.py", "w", encoding="utf-8") as f:
        f.writelines(out)
    print("SUCCESS: Patch 5 applied - elig_map built, game_view joined, _game_decision wired")
else:
    print(f"FAILED: found_elig={found_elig} found_decision={found_decision}")
