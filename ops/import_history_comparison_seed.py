"""Load a rebuildable comparison seed from byte-verified archived production inputs."""

import argparse
from collections import defaultdict
from contextlib import closing
from itertools import islice
import json
from pathlib import Path, PureWindowsPath
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import history_blocks as blocks
from history_store import FREE_DISK_RESERVE, INSERT_FIXTURE, connect, file_digest, packed, source_signature, start_evidence
import shutil


def batches(rows, size=256):
    while batch := list(islice(rows, size)):
        yield batch


def sources(seed, archive):
    accepted = []
    for original, expected, count in seed.execute("SELECT path,sha256,rows FROM sources WHERE sha256 IS NOT NULL"):
        name = PureWindowsPath(original).name
        if not name.startswith("snapshots-before-") or not name.endswith(".csv.gz"):
            raise RuntimeError(f"Unexpected history source {name}")
        path = archive / name
        signature = source_signature(path)
        if expected != file_digest(path) or signature != source_signature(path):
            raise RuntimeError(f"Production archive does not match the staged source: {name}")
        accepted.append((str(path), signature, expected, count))
    if not accepted:
        raise RuntimeError("Missing source manifest for the historical seed")
    return accepted


def import_seed(db, seed, archive):
    manifest = sources(seed, archive)
    revised = set()
    quotes = 0
    for batch in batches(seed.execute("SELECT * FROM quote_blocks ORDER BY sport,event_id,market,day")):
        fresh = []
        variants = []
        for row in batch:
            key = tuple(row[:4])
            previous = db.execute("SELECT payload FROM quote_blocks WHERE (sport,event_id,market,day)=(?,?,?,?)",
                                  key).fetchone()
            if previous is None:
                fresh.append(tuple(row))
                quotes += row[6]
            elif previous[0] != row[7]:
                seeded = {tuple(q) for q in blocks.decode(row[7])}
                existing = {tuple(q) for q in blocks.decode(previous[0])}
                if existing - seeded:
                    revised.add(key[:2])
                variants.extend((*key[:3], *q) for q in seeded - existing)
                quotes += len(seeded - existing)
        with db:
            db.executemany("INSERT INTO quote_blocks VALUES (?,?,?,?,?,?,?,?)", fresh)
            blocks.merge(db, variants, verify=True)
            db.executemany("INSERT OR IGNORE INTO comparison_pending VALUES (?,?)", {row[:2] for row in fresh})
    for batch in batches(seed.execute("SELECT sport,event_id,matchup,first_at,last_at,scheduled_at FROM fixtures")):
        for row in batch:
            current = db.execute("SELECT last_at FROM fixtures WHERE (sport,event_id)=(?,?)", row[:2]).fetchone()
            if current and current[0] > row[4]:
                revised.add(tuple(row[:2]))
        with db:
            db.executemany(INSERT_FIXTURE, batch)
    groups = defaultdict(list)
    for row in seed.execute("SELECT * FROM comparison_windows ORDER BY sport,event_id,market,minutes"):
        groups[row[0], row[3]].append(tuple(row))
    count = 0
    for key, windows in groups.items():
        if key in revised:
            continue
        actual = start_evidence(db, *key)
        valid = [row for row in windows if not actual or row[4] < actual]
        with db:
            db.execute("DELETE FROM comparison_windows WHERE (sport,event_id)=(?,?)", key)
            db.executemany("INSERT INTO comparison_windows VALUES (?,?,?,?,?,?,?,?)", valid)
            db.execute("DELETE FROM comparison_pending WHERE (sport,event_id)=(?,?)", key)
        count += len(valid)
    with db:
        db.executemany("INSERT OR IGNORE INTO comparison_pending VALUES (?,?)", revised)
        db.executemany("INSERT OR REPLACE INTO sources VALUES (?,?,?,?)", manifest)
        db.execute("INSERT OR REPLACE INTO jobs VALUES ('comparison_generation',datetime('now'))")
    return {"verified_sources": len(manifest), "additional_quotes": quotes,
            "ready_windows": count, "fixtures_to_rebuild": len(revised)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=Path)
    parser.add_argument("--root", type=Path, default=Path("/opt/red-fox-market-dynamics"))
    parser.add_argument("--state", type=Path, default=Path("/var/lib/redfox-history/private"))
    args = parser.parse_args()
    if shutil.disk_usage(args.state).free < FREE_DISK_RESERVE + args.seed.stat().st_size * 2:
        raise RuntimeError("The history seed would come too close to the protected disk reserve")
    with closing(sqlite3.connect(args.seed.resolve().as_uri() + "?mode=ro", uri=True)) as seed:
        if seed.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("Staged historical index failed its SQLite integrity check")
        with closing(connect(args.state)) as db:
            print(packed(import_seed(db, seed, args.root / "data" / "snapshot_archive")))


if __name__ == "__main__":
    main()
