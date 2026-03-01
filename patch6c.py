with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = """        _rs = _rs.rename(columns={"market": "market_display"})"""
new = """        if "market_display" in _rs.columns:
            _rs = _rs.drop(columns=["market_display"])
        _rs = _rs.rename(columns={"market": "market_display"})"""

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS")
else:
    print("FAILED: anchor not found")
