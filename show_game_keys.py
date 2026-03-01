from pathlib import Path
import re

s = Path("main.py").read_text(encoding="utf-8")

m = re.search(r"game_keys\s*=\s*\[.*?\]", s, re.S)
if not m:
    print("game_keys not found")
else:
    print("\nFOUND game_keys definition:\n")
    print(m.group(0))
