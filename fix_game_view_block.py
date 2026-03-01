from pathlib import Path

p = Path("main.py")
lines = p.read_text(encoding="utf-8").splitlines()

start = None
end = None

# find the corrupted "game_view =" line
for i,l in enumerate(lines):
    if l.strip() == "game_view =":
        start = i
        break

if start is None:
    raise SystemExit("FAILED: could not find 'game_view ='")

# find the next line that contains only ")"
for j in range(start, min(start+40, len(lines))):
    if lines[j].strip() == ")":
        end = j
        break

if end is None:
    raise SystemExit("FAILED: could not find closing ')' after game_view")

# Replace the whole broken block with a proper assignment
indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
lines[start:end+1] = [f"{indent}game_view = ("]

p.write_text("\n".join(lines), encoding="utf-8")

print(f"OK: repaired lines {start+1}-{end+1}")
