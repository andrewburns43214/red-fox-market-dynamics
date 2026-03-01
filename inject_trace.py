from pathlib import Path
p = Path("main.py")
s = p.read_text(encoding="utf-8")

anchor = '_d.to_csv("data/dashboard.csv", index=False)'
inject = '''
print("\\n===== FINAL WRITER DATAFRAME =====")
print("rows:", len(_d))
print("has market_display:", "market_display" in _d.columns)
print("sample:")
print(_d.head(5))
print("==================================\\n")
'''

s = s.replace(anchor, inject + "\n" + anchor)
p.write_text(s, encoding="utf-8")
print("OK: tracer injected")
