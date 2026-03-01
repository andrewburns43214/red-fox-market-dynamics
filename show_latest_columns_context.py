from pathlib import Path
import re

s = Path("main.py").read_text(encoding="utf-8")

start = s.find("METRICS TAP (SIDE LEVEL")
print(s[start:start+800])
