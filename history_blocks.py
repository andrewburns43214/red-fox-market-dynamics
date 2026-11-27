"""Lossless quote blocks partitioned by UTC date and indexed by fixture and market."""

from collections import defaultdict
from itertools import groupby
import hashlib
import json
import shutil
import zlib


FIELDS = ("captured_at", "side", "scheduled_at", "line", "price", "open_line",
          "open_price", "tickets", "money", "evidence")
SCHEMA = """
CREATE TABLE IF NOT EXISTS quote_blocks (
    sport TEXT NOT NULL, event_id TEXT NOT NULL, market TEXT NOT NULL,
    day TEXT NOT NULL, first_at TEXT NOT NULL, last_at TEXT NOT NULL,
    observations INTEGER NOT NULL, payload BLOB NOT NULL,
    PRIMARY KEY(sport,event_id,market,day)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS comparison_pending (
    sport TEXT NOT NULL, event_id TEXT NOT NULL, PRIMARY KEY(sport,event_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS comparison_windows (
    sport TEXT NOT NULL, market TEXT NOT NULL, minutes INTEGER NOT NULL,
    event_id TEXT NOT NULL, decision_at TEXT NOT NULL, entry_at TEXT NOT NULL,
    scheduled_at TEXT NOT NULL, evidence BLOB NOT NULL,
    PRIMARY KEY(sport,market,minutes,event_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS comparison_event ON comparison_windows(sport,event_id);
"""


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def encode(value):
    return zlib.compress(compact_json(value).encode("utf-8"), level=3)


def decode(payload):
    return json.loads(zlib.decompress(payload))


def logical_row(full):
    row = (*full[:5], *full[6:])
    derived = dict(zip(FIELDS, row[3:]))
    # Preserve legacy identities where SQLite normalized a value (for example
    # a signed zero). Ordinary fingerprints need no space in the archive.
    return row if full[5] == fingerprint(derived) else (*row, full[5])


def merge(db, rows, *, verify=False):
    grouped = defaultdict(set)
    for row in rows:
        grouped[(*row[:3], row[3][:10])].add(row[3:])
    dirty = set()
    for key, incoming in grouped.items():
        old = db.execute("SELECT payload FROM quote_blocks WHERE (sport,event_id,market,day)=(?,?,?,?)",
                         key).fetchone()
        existing = {tuple(row) for row in decode(old[0])} if old else set()
        exceptional = {row[-1] for row in existing if len(row) > len(FIELDS)}
        if exceptional:
            incoming = {row for row in incoming if len(row) > len(FIELDS) or
                        fingerprint(dict(zip(FIELDS, row))) not in exceptional}
        if incoming <= existing:
            continue
        merged = sorted(existing | incoming, key=lambda row: (row[:2], compact_json(row[2:])))
        payload = encode(merged)
        if verify and {tuple(row) for row in decode(payload)} != existing | incoming:
            raise RuntimeError(f"Historical quote round-trip failed for {key}")
        db.execute("INSERT OR REPLACE INTO quote_blocks VALUES (?,?,?,?,?,?,?,?)",
                   (*key, merged[0][0], merged[-1][0], len(merged), payload))
        dirty.add(key[:2])
    db.executemany("INSERT OR IGNORE INTO comparison_pending VALUES (?,?)", dirty)


def read_quotes(db, key, since="", until="9999"):
    blocks = db.execute("""
        SELECT payload FROM quote_blocks WHERE (sport,event_id,market)=(?,?,?)
          AND day BETWEEN ? AND ? AND last_at>=? AND first_at<=? ORDER BY day
    """, (*key, since[:10], until[:10], since, until))
    for block in blocks:
        for row in decode(block[0]):
            if since <= row[0] <= until:
                value = dict(zip(FIELDS, row)) | dict(zip(("sport", "event_id", "market"), key))
                if len(row) > len(FIELDS):
                    value["legacy_fingerprint"] = row[-1]
                yield value


def fingerprint(row):
    if legacy := row.get("legacy_fingerprint"):
        return legacy
    value = compact_json([row["scheduled_at"], row["line"], row["price"], row["open_line"],
                          row["open_price"], row["tickets"], row["money"], row["evidence"]])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def migrate(db, state, reserve, *, batch=128):
    """Resume checked migration by fixture without changing the source manifest."""
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='quotes'").fetchone()
    migrated = 0
    if exists:
        while shutil.disk_usage(state).free > reserve:
            saved = db.execute("SELECT value FROM jobs WHERE name='legacy_quote_cursor'").fetchone()
            cursor = tuple(json.loads(saved[0])) if saved else ("", "", "")
            keys = db.execute("""
                SELECT DISTINCT sport,event_id,market FROM quotes
                WHERE (sport,event_id,market)>(?,?,?) ORDER BY sport,event_id,market LIMIT ?
            """, (*cursor, batch)).fetchall()
            if not keys:
                # Disable zero-filling freed index pages. This data is a derived
                # copy and every logical row has passed the round-trip check.
                db.execute("PRAGMA secure_delete=OFF")
                with db:
                    db.execute("DROP TABLE quotes")
                    db.execute("DELETE FROM jobs WHERE name='legacy_quote_cursor'")
                    db.execute("INSERT OR REPLACE INTO jobs VALUES ('compact_vacuum_pending','1')")
                break
            for key in keys:
                raw = db.execute("""
                    SELECT * FROM quotes WHERE (sport,event_id,market)=(?,?,?)
                    ORDER BY captured_at,side,fingerprint
                """, tuple(key))
                with db:
                    for _, day in groupby(raw, key=lambda row: row["captured_at"][:10]):
                        rows = [logical_row(tuple(row)) for row in day]
                        merge(db, rows, verify=True)
                        migrated += len(rows)
                    db.execute("INSERT OR REPLACE INTO jobs VALUES ('legacy_quote_cursor',?)",
                               (compact_json(tuple(key)),))
        else:
            return {"migrated_quotes": migrated, "capacity_deferred": True}
    pending = db.execute("SELECT 1 FROM jobs WHERE name='compact_vacuum_pending'").fetchone()
    # VACUUM may temporarily require twice the pre-VACUUM database size.
    size = (state / "history.sqlite3").stat().st_size
    if pending and shutil.disk_usage(state).free >= reserve + size * 2:
        db.execute("VACUUM")
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("Historical database failed its post-compaction integrity check")
        with db:
            db.execute("DELETE FROM jobs WHERE name='compact_vacuum_pending'")
        pending = False
    return {"migrated_quotes": migrated, "vacuum_pending": bool(pending)}
