from pathlib import Path

p = Path("main.py")
lines = p.read_text(encoding="utf-8").splitlines()

start = None
end = None

for i, line in enumerate(lines):
    if line.startswith("def _timing_bucket_from_minutes"):
        start = i
    if start is not None and line.startswith("def cmd_snapshot"):
        end = i
        break

if start is None or end is None:
    raise SystemExit("Could not find block boundaries")

for i in range(start, end):
    lines[i] = "    " + lines[i]

p.write_text("\n".join(lines), encoding="utf-8")
print("OK: re-indented timing block inside build_dashboard")
