from pathlib import Path

p = Path("main.py")
lines = p.read_text(encoding="utf-8").splitlines()

start = 4985
end   = 5075

for i in range(start, end+1):
    if i-1 < len(lines):
        print(f"{i:04d}: {lines[i-1]}")
