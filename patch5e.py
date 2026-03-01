with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = """            state_map[k] = {
                "sport": sport,
                "game_id": game_id,
                "market": market,
                "side": side,"""

new = """            state_map[k] = {
                "sport": sport,
                "game_id": game_id,
                "market": market,
                "side": _side_norm,"""

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: state_map upsert now writes _side_norm")
else:
    print("FAILED: anchor not found")
