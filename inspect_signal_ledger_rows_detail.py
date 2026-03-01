import csv

ledger_file = "data/signal_ledger.csv"

with open(ledger_file, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    header_len = len(header)
    print(f"Header ({header_len} columns): {header}\n")
    
    malformed_rows = []
    for i, row in enumerate(reader, start=2):
        if len(row) != header_len:
            malformed_rows.append((i, len(row), row))
    
    if malformed_rows:
        print(f"Found {len(malformed_rows)} malformed rows:")
        for line_num, field_count, row_data in malformed_rows:
            print(f"Line {line_num}: {field_count} fields -> {row_data}")
    else:
        print("No malformed rows detected.")

    # Show first 5 rows for inspection
    f.seek(0)
    _ = next(reader)  # skip header again
    print("\nFirst 5 rows:")
    for i, row in enumerate(reader):
        print(row)
        if i >= 4:
            break
