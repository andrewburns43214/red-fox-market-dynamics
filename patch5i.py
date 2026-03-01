with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = """        # Also strip any residual "team" prefix left by prior double-normalization
        import re as _re2
        s = _re2.sub(r"^team(?=[a-z])", "", s).strip()"""

new = """        # Also strip any residual "team" prefix left by prior double-normalization
        import re as _re2
        while _re2.match(r"^team[a-z]", s):
            s = s[4:]"""

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS")
else:
    print("FAILED: anchor not found")
