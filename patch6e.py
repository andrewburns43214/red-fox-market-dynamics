with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = '            s = float(row.get("model_score", row.get("confidence_score", 0)))'
new = '            s = float(row.get("game_confidence", row.get("model_score", row.get("confidence_score", 0))))'

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS")
else:
    print("FAILED: anchor not found")
