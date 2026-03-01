from pathlib import Path
p = Path("main.py")
s = p.read_text(encoding="utf-8")

anchor = "_tmp = latest.copy()"

if anchor not in s:
    print("ANCHOR NOT FOUND — ABORT")
    raise SystemExit

inject = """
        print("\\n[AUDIT latest columns in FORCE WIDE]")
        print(list(_tmp.columns))
        print("[END AUDIT]\\n")
"""

s = s.replace(anchor, anchor + inject, 1)

p.write_text(s, encoding="utf-8")
print("OK: injected FORCE WIDE audit")
