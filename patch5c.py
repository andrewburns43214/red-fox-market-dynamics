with open("main.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

out = []
found = False
for i, line in enumerate(lines):
    if not found and '            k = f"{sport}|{game_id}|{market}|{side}"' in line:
        out.append("            # Use normalize_side_key for canonical row_state key\n")
        out.append("            _side_norm = normalize_side_key(sport, market, side)\n")
        out.append('            k = f"{sport}|{game_id}|{market}|{_side_norm}"\n')
        found = True
    else:
        out.append(line)

if found:
    with open("main.py", "w", encoding="utf-8") as f:
        f.writelines(out)
    print("SUCCESS: metrics tap key now uses normalize_side_key()")
else:
    print("FAILED: anchor not found")
