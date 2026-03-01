from pathlib import Path

p = Path("main.py")
lines = p.read_text(encoding="utf-8").splitlines()

# find build_dashboard start
start = None
for i, ln in enumerate(lines, start=1):
    if ln.startswith("def build_dashboard"):
        start = i
        break
if not start:
    raise SystemExit("build_dashboard not found")

# find next top-level def after build_dashboard
end = len(lines) + 1
for i in range(start, len(lines)):
    ln = lines[i]
    if ln.startswith("def ") and i+1 != start:
        end = i+1
        break

print(f"build_dashboard lines: {start}..{end-1}")

# print all returns inside that range + some context
for i in range(start, end):
    ln = lines[i-1]
    if ln.lstrip().startswith("return"):
        print("\n--- RETURN @ line", i, "---")
        lo = max(start, i-6)
        hi = min(end-1, i+6)
        for j in range(lo, hi+1):
            mark = ">>" if j == i else "  "
            print(f"{mark} {j}: {lines[j-1]}")
