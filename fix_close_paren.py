from pathlib import Path

p = Path("main.py")
lines = p.read_text(encoding="utf-8").splitlines()

insert_at = None

for i,l in enumerate(lines):
    if 'game_view["net_edge"]' in l:
        insert_at = i
        break

if insert_at is None:
    raise SystemExit("FAILED: could not locate net_edge anchor")

# Only insert if previous line isn't already ')'
if lines[insert_at-1].strip() != ")":
    indent = lines[insert_at][:len(lines[insert_at]) - len(lines[insert_at].lstrip())]
    lines.insert(insert_at, indent + ")")
    print("Inserted missing closing parenthesis")
else:
    print("Parenthesis already present")

p.write_text("\n".join(lines), encoding="utf-8")
