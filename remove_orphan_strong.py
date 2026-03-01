from pathlib import Path

p = Path("main.py")
lines = p.read_text(encoding="utf-8").splitlines()

start = None
end = None

# find the strong aggregation marker
for i,l in enumerate(lines):
    if "[v1.1 strong aggregation skipped]" in l:
        end = i
        break

if end is None:
    raise SystemExit("FAILED: cannot find strong block end")

# walk backwards to the try:
for j in range(end, max(end-40,0), -1):
    if "aggregate STRONG eligibility" in lines[j]:
        start = j-1
        break

if start is None:
    raise SystemExit("FAILED: cannot find strong block start")

# remove block
del lines[start:end+1]

# also remove the dangling conditional line if present
lines = [l for l in lines if "if 'game_view' in locals() else _pd.DataFrame()" not in l]

p.write_text("\n".join(lines), encoding="utf-8")

print(f"Removed lines {start+1} to {end+1}")
