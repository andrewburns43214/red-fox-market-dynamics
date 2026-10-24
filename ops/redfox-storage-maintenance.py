"""Bound Red Fox hot storage and reclaim crash debris."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publication_coverage import CoverageStore
from snapshot_store import compact_snapshot_history, snapshot_lock


def parse_time(value):
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except (TypeError, ValueError):
        return None


def compact_jsonl(path, timestamp_getter, cutoff, *, key_getter=None, max_bytes=128 * 1024 * 1024):
    """Stream-retain recent audit data; optionally keep only the latest key."""
    if not path.exists() or path.stat().st_size <= max_bytes:
        return 0
    temporary = path.with_name(f'.{path.name}.{__import__("os").getpid()}.compact.tmp')
    latest = {} if key_getter else None
    kept = 0
    try:
        with path.open(encoding='utf-8') as source, temporary.open('w', encoding='utf-8') as target:
            for line in source:
                try:
                    entry = json.loads(line)
                    captured = parse_time(timestamp_getter(entry))
                except (TypeError, ValueError):
                    continue
                if not captured or captured < cutoff:
                    continue
                if latest is not None:
                    key = key_getter(entry)
                    if key:
                        latest[key] = entry
                else:
                    target.write(json.dumps(entry, sort_keys=True, default=str, separators=(',', ':')) + '\n')
                    kept += 1
        if latest is not None:
            with temporary.open('w', encoding='utf-8') as target:
                for entry in latest.values():
                    target.write(json.dumps(entry, sort_keys=True, default=str, separators=(',', ':')) + '\n')
            kept = len(latest)
        temporary.replace(path)
        return kept
    finally:
        temporary.unlink(missing_ok=True)


def maintain(root, *, compact_snapshots=False, vacuum_coverage=False):
    data = root / 'data'
    snapshot = data / 'snapshots.csv'
    with snapshot_lock(snapshot):
        pass
    cutoff = datetime.now(timezone.utc) - timedelta(days=14)
    private = data / 'prop_projection'
    projection_time = lambda row: row.get('projection', {}).get('generated_at') or row.get('projection', {}).get('commence_time')
    projection_key = lambda row: f"{row.get('projection', {}).get('sport')}:{row.get('projection', {}).get('event_id')}"
    ledgers = {
        'projections': compact_jsonl(private / 'projection_ledger.jsonl', projection_time, cutoff, key_getter=projection_key),
        'shadow': compact_jsonl(private / 'projection_v2_shadow_ledger.jsonl', projection_time, cutoff, key_getter=projection_key),
        'observations': compact_jsonl(private / 'observations.jsonl', lambda row: row.get('captured_at'), cutoff),
    }
    store = CoverageStore(data)
    store.prune()
    if vacuum_coverage and store.path.exists():
        with sqlite3.connect(store.path, timeout=120) as db:
            db.execute('VACUUM')
    snapshots = None
    compact_bytes = int(os.environ.get("REDFOX_SNAPSHOT_COMPACT_BYTES", str(96 * 1024 * 1024)))
    if compact_snapshots or (snapshot.exists() and snapshot.stat().st_size > compact_bytes):
        snapshots = compact_snapshot_history(snapshot)
    return {'ledgers': ledgers, 'snapshots': snapshots}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/opt/red-fox-market-dynamics')
    parser.add_argument('--compact-snapshots', action='store_true')
    parser.add_argument('--vacuum-coverage', action='store_true')
    args = parser.parse_args()
    print(json.dumps(maintain(Path(args.root), compact_snapshots=args.compact_snapshots, vacuum_coverage=args.vacuum_coverage), sort_keys=True))


if __name__ == '__main__':
    main()
