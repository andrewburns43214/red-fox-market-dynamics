from pathlib import Path

p = Path("main.py")
lines = p.read_text(encoding="utf-8").splitlines()

target = "update_row_state_and_signal_ledger(_metrics_df)"
out = []

inserted = False

for line in lines:
    if target in line and not inserted:
        indent = line[:len(line) - len(line.lstrip())]

        out.append(indent + 'print("[METRICS PROBE] rows=", len(_metrics_df))')
        out.append(indent + 'print("[METRICS PROBE] cols=", list(_metrics_df.columns)[:20])')
        out.append(indent + 'print("[METRICS PROBE] sample=", _metrics_df.head(2).to_dict("records"))')

        inserted = True

    out.append(line)

if not inserted:
    raise SystemExit("FAILED — metrics callsite not found")

p.write_text("\n".join(out), encoding="utf-8")
print("[ok] safe probe injected")
