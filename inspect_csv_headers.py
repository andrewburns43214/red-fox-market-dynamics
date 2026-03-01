from pathlib import Path
import hashlib

FILES = [
    "data/snapshots.csv",
    "data/dashboard.csv",
    "data/row_state.csv",
    "data/signal_ledger.csv",
]

def header_fingerprint(path):
    raw = Path(path).read_bytes()
    # take only first line
    first_line = raw.splitlines()[0] if raw.splitlines() else b""
    # normalize common header junk (BOM, CR)
    line = first_line.replace(b"\xef\xbb\xbf", b"").replace(b"\r", b"")
    return hashlib.sha1(line).hexdigest(), line.decode("utf-8", errors="replace")

for f in FILES:
    p = Path(f)
    print("\n=== ", f, " ===")
    if not p.exists():
        print("MISSING")
        continue
    h, line = header_fingerprint(f)
    print("sha1(header)=", h)
    print("header_line=", line)
