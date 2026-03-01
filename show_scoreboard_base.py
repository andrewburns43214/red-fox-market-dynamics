from pathlib import Path

p = Path("main.py")
lines = p.read_text(encoding="utf-8").splitlines()

for i,l in enumerate(lines):
    if "ESPN_SCOREBOARD_BASE" in l:
        print(f"{i+1:04d}: {l}")
