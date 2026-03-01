import subprocess, shutil, pathlib, sys

bak = pathlib.Path("bak")
main = pathlib.Path("main.py")
tmp  = pathlib.Path("main_test.py")

candidates = sorted(bak.glob("main.py.*.ps1safe.py"), key=lambda p: p.stat().st_mtime, reverse=True)

print(f"Testing {len(candidates)} backups...\n")

for f in candidates:
    shutil.copy(f, tmp)

    # compile test
    r = subprocess.run([sys.executable, "-m", "py_compile", str(tmp)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if r.returncode != 0:
        print("COMPILE FAIL:", f.name)
        continue

    # runtime test
    r = subprocess.run([sys.executable, str(tmp), "report"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if r.returncode == 0:
        print("\nFOUND WORKING BUILD:", f.name)
        main.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")
        print("RESTORED -> main.py")
        sys.exit(0)
    else:
        print("RUNTIME FAIL:", f.name)

print("\nNO WORKING BACKUP FOUND")
sys.exit(1)
