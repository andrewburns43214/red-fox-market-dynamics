"""Independent historical quote, decision and settlement index."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta, timezone
from itertools import groupby
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time

import history_blocks as blocks

SPORTS = {"nfl", "ncaaf", "nba", "ncaab", "nhl", "mlb", "ufc"}
FREE_DISK_RESERVE = 15 * 1024**3
PRICE = re.compile(r"(?:@\s*|\()([+-]\d{3,})\s*\)?$")
HANDICAP = re.compile(r"\s([+-]?\d+(?:\.\d+)?)$")
PICKEM = re.compile(r"\s(?:PK|PICK|PICK'EM)$", re.IGNORECASE)
MOVE_FIELDS = (
    "reaction", "path", "data_badge", "response_direction", "evidence_role",
    "evidence_polarity", "max_excursion", "return_toward_open", "line_dir_changes",
    "material_direction_changes", "active_worsening_reversal", "whipsaw_retention_ratio",
    "whipsaw_recovered", "whipsaw_confirmation_ready", "context_chips",
)
BOARD_FIELDS = (
    "market_rationale", "supported_side", "reaction", "path", "data_badge",
    "cross_market_mismatch", "cross_market_split", "cross_market_opener_mismatch_verified",
    "favorite_state", "favorite_reason", "favorite_side", "favorite_rule_version",
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    path TEXT PRIMARY KEY, signature TEXT NOT NULL, sha256 TEXT, rows INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS fixtures (
    sport TEXT NOT NULL, event_id TEXT NOT NULL, matchup TEXT NOT NULL,
    first_at TEXT NOT NULL, last_at TEXT NOT NULL, scheduled_at TEXT,
    PRIMARY KEY (sport,event_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS decisions (
    sport TEXT NOT NULL, event_id TEXT NOT NULL, market TEXT NOT NULL,
    known_at TEXT NOT NULL, source TEXT NOT NULL, digest TEXT NOT NULL, evidence TEXT NOT NULL,
    PRIMARY KEY (sport,event_id,market,known_at,source,digest)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS board_state (
    sport TEXT NOT NULL, event_id TEXT NOT NULL, market TEXT NOT NULL, digest TEXT NOT NULL,
    PRIMARY KEY (sport,event_id,market)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS outcomes (
    sport TEXT NOT NULL, event_id TEXT NOT NULL, source TEXT NOT NULL,
    observed_at TEXT NOT NULL, status TEXT NOT NULL,
    provider_event_id TEXT, evidence TEXT NOT NULL, digest TEXT NOT NULL,
    PRIMARY KEY (sport,event_id,source,digest)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS outcomes_provider ON outcomes(sport,source,provider_event_id);
CREATE TABLE IF NOT EXISTS unresolved (
    sport TEXT NOT NULL, event_id TEXT NOT NULL, status TEXT NOT NULL,
    checked_at TEXT NOT NULL, detail TEXT NOT NULL, PRIMARY KEY(sport,event_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS jobs (
    name TEXT PRIMARY KEY, value TEXT NOT NULL
);
"""


def utc(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds") if parsed.tzinfo else None


def now_iso():
    return utc(datetime.now(timezone.utc))


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def connect(state):
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    db = sqlite3.connect(state / "history.sqlite3", timeout=20)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA + blocks.SCHEMA)
    return db


@contextmanager
def atomic_text(path, text):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def number(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def quote_part(quote):
    match = PRICE.search(quote or "")
    if not match:
        return None, None, ""
    label = quote[:match.start()].strip()
    line = HANDICAP.search(label)
    if pick := PICKEM.search(label):
        return 0.0, int(match[1]), label[:pick.start()]
    return (float(line[1]), int(match[1]), label[:line.start()]) if line else (
        None, int(match[1]), label
    )


def quote_row(row):
    sport, event = row.get("sport"), row.get("game_id")
    captured = utc(row.get("timestamp"))
    side = row.get("side") or ""
    current, price, label = quote_part(row.get("current_line"))
    opening, open_price, _ = quote_part(row.get("open_line"))
    if sport not in SPORTS or not event or not captured or not row.get("game") or price is None:
        return None
    market = "TOTAL" if side.startswith(("Over ", "Under ")) else (
        "SPREAD" if HANDICAP.search(side) or PICKEM.search(side) or str(row.get("market", "")).lower() == "spread"
        else "MONEYLINE"
    )
    if sport == "ufc" and market != "MONEYLINE":
        return None
    identity = label.casefold().strip()
    if not identity or current is None and market != "MONEYLINE":
        return None
    scheduled = utc(row.get("dk_start_iso"))
    tickets, money = number(row.get("bets_pct")), number(row.get("money_pct"))
    evidence = packed({key: row.get(key) for key in (
        "game", "side", "market", "open_line", "current_line", "injury_news", "key_number_note"
    ) if row.get(key)})
    return (
        sport, event, market, captured, identity, scheduled,
        current, price, opening, open_price, tickets, money, evidence,
    )


INSERT_FIXTURE = """
INSERT INTO fixtures VALUES (?,?,?,?,?,?)
ON CONFLICT(sport,event_id) DO UPDATE SET
    first_at=min(fixtures.first_at,excluded.first_at),
    matchup=CASE WHEN excluded.last_at>=fixtures.last_at THEN excluded.matchup ELSE fixtures.matchup END,
    scheduled_at=CASE WHEN excluded.last_at>=fixtures.last_at
                  THEN coalesce(excluded.scheduled_at,fixtures.scheduled_at) ELSE fixtures.scheduled_at END,
    last_at=max(fixtures.last_at,excluded.last_at)
"""


def source_signature(path):
    st = path.stat()
    return f"{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"


def ingest_quotes(db, path, *, immutable=False, seconds=None):
    """Deduplicate exact observations and retain conflicting versions for review."""
    sig = source_signature(path)
    old = db.execute("SELECT signature,sha256 FROM sources WHERE path=?", (str(path),)).fetchone()
    if old and old["signature"] == sig:
        return 0
    if immutable and old and old["sha256"] and file_digest(path) != old["sha256"]:
        raise RuntimeError(f"Indexed immutable history changed: {path.name}")
    enter = time.monotonic()
    watermark = None
    if not immutable:
        latest = db.execute("SELECT value FROM jobs WHERE name='hot_watermark'").fetchone()
        if latest:
            watermark = utc(datetime.fromisoformat(latest[0]) - timedelta(hours=1))
    read = gzip.open if path.suffix == ".gz" else open
    cursor_key = f"archive_cursor:{path}" if immutable and seconds is not None else None
    prior = db.execute("SELECT value FROM jobs WHERE name=?", (cursor_key,)).fetchone() if cursor_key else None
    progress = json.loads(prior[0]) if prior else {}
    if progress.get("signature") != sig:
        progress = {}
    accepted = progress.get("rows", 0)
    fixtures = {}
    quote_buffer = []
    with db, read(path, "rt", newline="", encoding="utf-8-sig") as handle:
        # readline keeps text seek cookies available after logical CSV
        # records, including records with quoted embedded newlines.
        reader = csv.DictReader(iter(handle.readline, ""))
        if not {"timestamp", "sport", "game_id", "current_line"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"Unrecognized archive header: {path.name}")
        if progress:
            handle.seek(progress["offset"])
        for raw in reader:
            if watermark and (captured := utc(raw.get("timestamp"))) and captured < watermark:
                continue
            row = quote_row(raw)
            if not row:
                continue
            accepted += 1
            quote_buffer.append(row)
            event = row[:2]
            capture = row[3]
            if event not in fixtures:
                fixtures[event] = (raw["game"], capture, capture, row[5])
            else:
                game, first, last, start = fixtures[event]
                fixtures[event] = (
                    raw["game"] if capture >= last else game, min(first, capture), max(last, capture),
                    (row[5] or start) if capture >= last else start,
                )
            if len(quote_buffer) == 4000:
                blocks.merge(db, quote_buffer)
                quote_buffer.clear()
                if cursor_key:
                    db.executemany(INSERT_FIXTURE, ((*key, *value) for key, value in fixtures.items()))
                    fixtures.clear()
                    db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)", (cursor_key, packed({
                        "signature": sig, "offset": handle.tell(), "rows": accepted,
                    })))
                db.commit()
                if seconds and time.monotonic() - enter > seconds:
                    raise TimeoutError(f"Quote ingest exceeded time budget: {path.name}")
        blocks.merge(db, quote_buffer)
        db.executemany(INSERT_FIXTURE, ((*key, *value) for key, value in fixtures.items()))
        if not immutable and fixtures:
            newest = max(value[2] for value in fixtures.values())
            db.execute("INSERT INTO jobs VALUES ('hot_watermark',?) "
                       "ON CONFLICT(name) DO UPDATE SET value=max(value,excluded.value)", (newest,))
        if source_signature(path) == sig:
            db.execute("INSERT OR REPLACE INTO sources VALUES (?,?,?,?)",
                       (str(path), sig, file_digest(path) if immutable else None, accepted))
            if cursor_key:
                db.execute("DELETE FROM jobs WHERE name=?", (cursor_key,))
    return accepted


def save_decisions(db, rows):
    with db:
        db.executemany("INSERT OR IGNORE INTO decisions VALUES (?,?,?,?,?,?,?)", rows)


def ingest_decisions(db, path):
    sig = source_signature(path)
    prior = db.execute("SELECT signature FROM sources WHERE path=?", (str(path),)).fetchone()
    if prior and prior["signature"] == sig:
        return 0
    count = 0
    batch = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for r in csv.DictReader(handle):
            sport, event = r.get("sport"), r.get("game_id")
            clock = utc(r.get("recorded_at") or r.get("reviewed_at") or r.get("captured_at_utc"))
            if sport not in SPORTS or not event or not clock:
                continue
            value = packed(r)
            batch.append((sport, event, r.get("market_display", ""), clock, path.stem, digest(value), value))
            count += 1
            if len(batch) == 250:
                save_decisions(db, batch)
                batch.clear()
    save_decisions(db, batch)
    with db:
        if source_signature(path) == sig:
            db.execute("INSERT OR REPLACE INTO sources VALUES (?,?,?,?)", (str(path), sig, None, count))
    return count


def ingest_board(db, path):
    """Store changes in public evidence with a publication-time knowledge clock."""
    sig = source_signature(path)
    prior = db.execute("SELECT signature FROM sources WHERE path=?", (str(path),)).fetchone()
    if prior and prior["signature"] == sig:
        return 0
    known_at = utc(datetime.fromtimestamp(path.stat().st_mtime, timezone.utc))
    count = 0
    with db, path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sport, event, market = (row.get(field) for field in ("sport", "game_id", "market_display"))
            if sport not in SPORTS or not event or not market:
                continue
            try:
                sides = json.loads(row.get("market_sides") or "[]")
            except ValueError:
                continue
            evidence = {key: row[key] for key in BOARD_FIELDS if row.get(key)}
            evidence["sides"] = [
                {key: side[key] for key in ("flagged_side", *MOVE_FIELDS) if key in side}
                for side in sides if isinstance(side, dict)
            ]
            value = packed(evidence)
            token = digest(value)
            key = sport, event, market
            prior = db.execute("SELECT digest FROM board_state WHERE sport=? AND event_id=? AND market=?", key).fetchone()
            if prior and prior["digest"] == token:
                continue
            evidence["source_times"] = [side.get("source_latest_at") for side in sides if isinstance(side, dict)]
            db.execute("INSERT OR IGNORE INTO decisions VALUES (?,?,?,?,?,?,?)",
                       (*key, known_at, "published_board", token, packed(evidence)))
            db.execute("INSERT OR REPLACE INTO board_state VALUES (?,?,?,?)", (*key, token))
            count += 1
        if source_signature(path) != sig:
            raise RuntimeError("Board changed during archive read. Retry on next run.")
        db.execute("INSERT OR REPLACE INTO sources VALUES (?,?,?,?)", (str(path), sig, None, count))
    return count


def ingest_coverage(db, path):
    """Retain dated publication evidence without copying full rendered HTML."""
    old = db.execute("SELECT value FROM jobs WHERE name='coverage_watermark'").fetchone()
    start = old[0] if old else ""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5) as coverage:
        records = coverage.execute("""
          SELECT sport,game_id,market_display,recorded_at,state,payload
          FROM transitions WHERE recorded_at>=? ORDER BY recorded_at
        """, (start,)).fetchall()
    batch = []
    for sport, event, market, observed, state, payload in records:
        known_at = utc(observed)
        if sport not in SPORTS or not event or not known_at:
            continue
        value = packed({"state": state, "source": json.loads(payload)})
        batch.append((sport, event, market, known_at, "publication_transition", digest(value), value))
        if len(batch) == 250:
            save_decisions(db, batch)
            batch.clear()
    save_decisions(db, batch)
    with db:
        if records:
            db.execute("INSERT OR REPLACE INTO jobs VALUES ('coverage_watermark',?)", (records[-1][3],))
    return len(records)


def health(db, root, state):
    shards = db.execute("SELECT path,sha256 FROM sources WHERE sha256 IS NOT NULL ORDER BY path").fetchall()
    if shards:
        day = int(time.time() // 86400)
        for position in range(min(12, len(shards))):
            row = shards[(day * 12 + position) % len(shards)]
            if not Path(row["path"]).is_file() or file_digest(Path(row["path"])) != row["sha256"]:
                raise RuntimeError(f"Historical archive integrity failure: {Path(row['path']).name}")
    disk = shutil.disk_usage(root)
    # Independently decode a rotating sample of blocks as well as hashing raw sources.
    for recorded, payload in db.execute("""
        SELECT observations,payload FROM quote_blocks
        WHERE day>=date('now', '-' || (abs(random()) % 360) || ' days') LIMIT 12
    """):
        if len(blocks.decode(payload)) != recorded:
            raise RuntimeError("Historical quote block row count changed")
    summary = {"at": now_iso(), "disk_total": disk.total, "disk_available": disk.free,
               "source_files": len(shards), "index_bytes": (state / "history.sqlite3").stat().st_size,
               "capacity_warning": disk.free < FREE_DISK_RESERVE}
    summary["quote_blocks"], summary["quotes"], summary["quote_payload_bytes"] = db.execute("""
        SELECT count(*),coalesce(sum(observations),0),coalesce(sum(length(payload)),0) FROM quote_blocks
    """).fetchone()
    summary["comparison_windows"] = db.execute("SELECT count(*) FROM comparison_windows").fetchone()[0]
    with atomic_text(state / "health.json", packed(summary)):
        pass
    return summary


def start_evidence(db, sport, event):
    starts = []
    for state, raw in db.execute(
        "SELECT status,evidence FROM outcomes WHERE sport=? AND event_id=?", (sport, event)
    ):
        value = json.loads(raw)
        clock = utc(value.get("actual_start") or value.get("first_pitch"))
        if sport == "ufc" and state in {"verified", "push", "void"} and value.get("start_source"):
            clock = clock or utc(value.get("start"))
        if clock:
            starts.append(clock)
    return min(starts, default=None)


def eligible_pairs(db, sport, event, market, until, since="", *, actual_start=None):
    if sport == "ufc" and actual_start is None:
        return
    rows = blocks.read_quotes(db, (sport, event, market), since, until)
    for capture, group in groupby(rows, key=lambda row: row["captured_at"]):
        if actual_start is not None and capture >= actual_start:
            break
        pair = tuple(group)
        if len(pair) != 2 or pair[0]["side"] == pair[1]["side"]:
            continue
        scheduled = pair[0]["scheduled_at"]
        if scheduled and scheduled == pair[1]["scheduled_at"] and capture < scheduled:
            yield capture, pair


def paired_as_of(db, sport, event, market, decision_at, since):
    """Return identifiable pre-start pairs known by the decision clock."""
    start, end = utc(since), utc(decision_at)
    if not start or not end:
        return []
    actual = start_evidence(db, sport, event)
    return [{key: value for key, value in row.items() if key != "legacy_fingerprint"} |
            {"fingerprint": blocks.fingerprint(row)}
            for _, pair in eligible_pairs(db, sport, event, market, end, start, actual_start=actual)
            for row in pair]


def store_outcome(db, sport, event, source, status, evidence, *, correction=False):
    value = packed(evidence)
    identity = {key: val for key, val in evidence.items() if key != "observed_at"}
    fingerprint = digest(packed(identity))
    if correction:
        prior = db.execute("""
            SELECT status,evidence FROM outcomes WHERE (sport,event_id,source)=(?,?,?)
            ORDER BY observed_at DESC,digest LIMIT 1
        """, (sport, event, source)).fetchone()
        if prior:
            original = json.loads(prior[1])
            original.pop("observed_at", None)
            if prior[0] == status and original == identity:
                return
            # Retain each confirmed revision, including a correction back
            # to an earlier score. Historical cutoffs still see the old row.
            fingerprint = digest(packed((status, evidence)))
    inserted = db.execute("INSERT OR IGNORE INTO outcomes VALUES (?,?,?,?,?,?,?,?)", (
        sport, event, source, utc(evidence.get("observed_at")) or now_iso(), status,
        evidence.get("provider_event_id"), value, fingerprint,
    )).rowcount
    if inserted:
        db.execute("INSERT OR IGNORE INTO comparison_pending VALUES (?,?)", (sport, event))


def latest_provider_outcomes(rows, cutoff):
    """Keep the latest known revision for each source/verified identity."""
    latest = {}
    for row in rows:
        source, _, provider, observed, _ = row
        if observed <= cutoff and (source, provider) not in latest:
            latest[source, provider] = row
    return list(latest.values())


def import_outcomes(db, path):
    """Accept attributable, pre-reconciled research evidence in a portable format."""
    count = 0
    with db, path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sport") not in SPORTS or not row.get("event_id") or not row.get("source"):
                raise ValueError(f"Incomplete outcome evidence in {path.name}")
            if row.get("status") not in {"verified", "void", "push", "unresolved", "unavailable"}:
                raise ValueError(f"Unsupported outcome status in {path.name}")
            store_outcome(db, row["sport"], row["event_id"], row["source"], row["status"], row)
            count += 1
    return count


def restore_evidence(db, source):
    """Load a backup's private evidence exports into a new isolated store."""
    counts = {}
    with db:
        for table in ("decisions", "outcomes", "unresolved"):
            columns = [row[1] for row in db.execute(f"PRAGMA table_info({table})")]
            placeholders = ",".join("?" for _ in columns)
            with (source / f"{table}.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    if set(record) != set(columns):
                        raise ValueError(f"Invalid {table} evidence record")
                    db.execute(f"INSERT INTO {table} VALUES ({placeholders})",
                               tuple(record[column] for column in columns))
            counts[table] = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return counts


def archive_sources(root, state):
    data = root / "data"
    known = [
        *sorted((data / "snapshot_archive").glob("snapshots-before-*.csv.gz"), reverse=True),
        data / "snapshots_PREBACKFILL.csv",
        data / "snapshots.bak_before_movement.csv",
        data / "_pre_v12_archive" / "snapshots.PRE_V12_BACKUP.csv",
        data / "v2_backup_epoch" / "snapshots.csv",
        *sorted(data.glob("snapshots.csv.bak_*")),
        *sorted((state / "research").glob("quotes-*.csv.gz")),
    ]
    return [path for path in known if path.is_file()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/opt/red-fox-market-dynamics"))
    parser.add_argument("--state", type=Path, default=Path("/var/lib/redfox-history/private"))
    parser.add_argument("command", choices=("archive", "compact", "observe", "decisions",
                                            "import-results", "restore-evidence", "summary", "hot", "health"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--max-shards", type=int, default=64)
    parser.add_argument("--archive-seconds", type=int, default=420)
    options = parser.parse_args()
    if options.command == "restore-evidence":
        if not options.input or (options.state / "history.sqlite3").exists():
            parser.error("restore-evidence needs --input and a new isolated --state directory")
    data = options.root / "data"
    with connect(options.state) as db:
        if options.command == "compact":
            print(packed(blocks.migrate(db, options.state, FREE_DISK_RESERVE)))
        elif options.command == "archive":
            seen = 0
            capacity_deferred, budget_deferred = False, False
            end = time.monotonic() + options.archive_seconds
            for path in archive_sources(options.root, options.state):
                if shutil.disk_usage(options.state).free < FREE_DISK_RESERVE:
                    capacity_deferred = True
                    break
                remaining = end - time.monotonic()
                if remaining <= 0:
                    budget_deferred = True
                    break
                try:
                    if ingest_quotes(db, path, immutable=True, seconds=remaining):
                        seen += 1
                except TimeoutError:
                    budget_deferred = True
                    break
                if seen >= options.max_shards:
                    break
            print(packed({"imported_shards": seen, "capacity_deferred": capacity_deferred,
                          "budget_deferred": budget_deferred}))
        elif options.command == "observe":
            print(packed({"decision_changes": ingest_board(db, data / "anomaly_board.csv")}))
        elif options.command == "decisions":
            sources = ("red_fox_favorite_tracking.csv", "red_fox_favorite_review.csv", "anomaly_action_ledger.csv")
            counts = {name: ingest_decisions(db, data / name) for name in sources if (data / name).exists()}
            if (data / "publication_coverage.sqlite3").is_file():
                counts["publication_transitions"] = ingest_coverage(db, data / "publication_coverage.sqlite3")
            print(packed(counts))
        elif options.command == "hot":
            if shutil.disk_usage(options.state).free < FREE_DISK_RESERVE:
                print(packed({"indexed_rows": 0, "deferred": "disk reserve"}))
            else:
                print(packed({"indexed_rows": ingest_quotes(db, data / "snapshots.csv")}))
        elif options.command == "import-results":
            print(packed({"evidence_rows": import_outcomes(db, options.input)}))
        elif options.command == "restore-evidence":
            print(packed(restore_evidence(db, options.input)))
        elif options.command == "health":
            summary = health(db, options.root, options.state)
            print(packed(summary))
            if summary["capacity_warning"]:
                raise SystemExit("Historical storage early warning: less than 15 GiB free.")
        else:
            counts = db.execute("""
                SELECT sport,market,sum(observations) AS rows,count(DISTINCT event_id) AS events,
                       min(first_at) AS first,max(last_at) AS last
                FROM quote_blocks GROUP BY sport,market
            """).fetchall()
            print(packed([dict(r) for r in counts]))


if __name__ == "__main__":
    main()
