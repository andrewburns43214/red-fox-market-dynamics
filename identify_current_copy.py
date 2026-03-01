import hashlib, glob, os

def sha1(path):
    with open(path,"rb") as f:
        return hashlib.sha1(f.read()).hexdigest()

current = sha1("main.py")
print("\nCURRENT main.py SHA1:", current, "\n")

matches = []

for f in glob.glob("bak/main.py*.ps1safe.py"):
    try:
        h = sha1(f)
        if h == current:
            matches.append(f)
    except:
        pass

if matches:
    print("EXACT MATCH FOUND:\n")
    for m in matches:
        print(m)
else:
    print("No identical backup found — this is a modified working copy.")
