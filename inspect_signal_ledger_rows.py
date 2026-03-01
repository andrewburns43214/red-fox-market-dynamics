import csv

with open("data/signal_ledger.csv", newline='') as f:
    reader = csv.reader(f)
    header_len = len(next(reader))
    for i, row in enumerate(reader, start=2):
        if len(row) != header_len:
            print(f"Line {i} malformed: {len(row)} fields")
