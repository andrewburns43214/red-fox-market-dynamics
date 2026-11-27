import csv
from datetime import datetime, timezone
import gzip
import importlib.util
import json
import os
from pathlib import Path
import time

import pytest

from history_store import connect, import_outcomes, ingest_board, ingest_quotes, paired_as_of, store_outcome
from history_store import quote_row
from history_blocks import fingerprint, FIELDS, migrate
from history_comparisons import candidates, rebuild
from history_outcomes import resolve_mlb
from mlb_fixtures import team_key


def write_quotes(path, rows):
    fields = ["timestamp", "sport", "game_id", "game", "side", "market", "open_line",
              "current_line", "bets_pct", "money_pct", "dk_start_iso"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pair(clock):
    common = dict(timestamp=clock, sport="nfl", game_id="one", game="BUF Bills @ KC Chiefs",
                  market="splits", dk_start_iso="2026-11-27T01:20:00Z")
    return [
        dict(common, side="BUF Bills +3", open_line="BUF Bills +3.5 @ -108",
             current_line="BUF Bills +3 @ -110", bets_pct=40, money_pct=35),
        dict(common, side="KC Chiefs -3", open_line="KC Chiefs -3.5 @ -112",
             current_line="KC Chiefs -3 @ -110", bets_pct=60, money_pct=65),
    ]


def test_archive_deduplication_conflicts_and_as_of(tmp_path):
    early = "2026-11-27T00:30:00Z"
    late = "2026-11-27T00:50:00Z"
    plain = tmp_path / "snapshots.csv"
    write_quotes(plain, pair(early) + pair(late))
    gz = tmp_path / "backup.csv.gz"
    gz.write_bytes(gzip.compress(plain.read_bytes()))
    with connect(tmp_path / "state") as db:
        assert ingest_quotes(db, gzip_path := gz, immutable=True) == 4
        assert ingest_quotes(db, gzip_path, immutable=True) == 0
        assert ingest_quotes(db, plain) == 4
        assert db.execute("SELECT sum(observations) FROM quote_blocks").fetchone()[0] == 4
        rows = paired_as_of(db, "nfl", "one", "SPREAD", "2026-11-27T00:40:00Z", early)
        assert [r["price"] for r in rows] == [-110, -110]
        assert [r["line"] for r in rows] == [3, -3]
        assert [r["open_line"] for r in rows] == [3.5, -3.5]
        store_outcome(db, "nfl", "one", "fixture", "verified", {"actual_start": "2026-11-27T00:45:00Z"})
        rows = paired_as_of(db, "nfl", "one", "SPREAD", "2026-11-27T01:00:00Z", early)
        assert len(rows) == 2 and rows[0]["captured_at"].startswith("2026-11-27T00:30")
        conflicting = pair(early)[0] | {"current_line": "BUF Bills +3 @ -115"}
        write_quotes(plain, pair(early) + pair(late) + [conflicting])
        ingest_quotes(db, plain)
        assert db.execute("SELECT sum(observations) FROM quote_blocks").fetchone()[0] == 5
        assert paired_as_of(db, "nfl", "one", "SPREAD", "2026-11-27T00:40:00Z", early) == []


def test_compact_migration_preserves_variants_and_reclaims_legacy_pages(tmp_path):
    early = "2026-11-26T23:59:00Z"
    late = "2026-11-27T00:40:00Z"
    original = pair(early) + pair(late)
    variant = pair(late)[0] | {"current_line": "BUF Bills +3 @ -115"}
    with connect(tmp_path) as db:
        db.execute("""
            CREATE TABLE quotes(sport,event_id,market,captured_at,side,fingerprint,scheduled_at,
                                line REAL,price INTEGER,open_line REAL,open_price INTEGER,
                                tickets REAL,money REAL,evidence,
                                PRIMARY KEY(sport,event_id,market,captured_at,side,fingerprint))
            WITHOUT ROWID
        """)
        for r in original + [variant]:
            row = quote_row(r)
            identifier = fingerprint(dict(zip(FIELDS, row[3:])))
            db.execute("INSERT INTO quotes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (*row[:5], identifier, *row[5:]))
        db.commit()
        assert migrate(db, tmp_path, 0, batch=1) == {"migrated_quotes": 5, "vacuum_pending": False}
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='quotes'").fetchone()
        assert db.execute("SELECT sum(observations) FROM quote_blocks").fetchone()[0] == 5
        rows = paired_as_of(db, "nfl", "one", "SPREAD", late, early)
        assert [(r["side"], r["line"], r["price"]) for r in rows] == [
            ("buf bills", 3.0, -110), ("kc chiefs", -3.0, -110)
        ]
        assert migrate(db, tmp_path, 0)["migrated_quotes"] == 0


def test_comparison_windows_use_only_known_inputs_and_one_verified_fixture(tmp_path):
    pre = "2026-11-27T00:48:00Z"
    after_decision = "2026-11-27T00:55:00Z"
    after_start = "2026-11-27T01:14:00Z"
    rows = pair(pre) + [dict(row, current_line=row["current_line"].replace("-110", "-119"))
                        for row in pair(after_decision) + pair(after_start)]
    ml = [
        dict(row, side=team, open_line=f"{team} @ {opening:+}",
             current_line=f"{team} @ {current:+}")
        for row, (team, opening, current) in zip(pair(pre), (("BUF Bills", 150, 130), ("KC Chiefs", -180, -140)))
    ]
    tape = tmp_path / "snapshots.csv"
    write_quotes(tape, rows + ml + [dict(row, game_id="duplicate") for row in rows + ml])
    now = datetime(2026, 11, 28, tzinfo=timezone.utc)
    with connect(tmp_path / "state") as db:
        ingest_quotes(db, tape)
        for key in ("one", "duplicate"):
            store_outcome(db, "nfl", key, "espn", "verified", {
                "provider_event_id": "same-fixture", "matchup": "BUF Bills @ KC Chiefs",
                "actual_start": "2026-11-27T01:12:00Z", "observed_at": "2026-11-27T05:00:00Z",
                "score_away": 21, "score_home": 24,
            })
        db.commit()
        assert rebuild(db, now, 20)["fixtures"] == 2
        cohort = list(candidates(db, "nfl", "SPREAD", 30, now))
        assert len(cohort) == 1
        entry = cohort[0]
        assert entry["entry_at"] == entry["cross_entry_at"] == "2026-11-27T00:48:00.000000+00:00"
        assert entry["decision_at"] == "2026-11-27T00:50:00.000000+00:00"
        assert [row["price"] for row in entry["sides"]] == [-110, -110]
        assert [row["result"] for row in entry["sides"]] == ["push", "push"]
        assert entry["sides"][0]["cross_market"]["price_toward_implied"] > 0
        assert list(candidates(db, "nfl", "SPREAD", 10, now))  # 01:10 predates verified 01:12 start.
        assert not list(candidates(db, "nfl", "SPREAD", 30, "2026-11-27T02:00:00Z"))


def test_nonfavorite_public_states_and_research_outcomes(tmp_path):
    board = tmp_path / "anomaly_board.csv"
    data = {
        "sport": "ncaab", "game_id": "two", "market_display": "SPREAD",
        "supported_side": "Baylor +3.5", "favorite_state": "not_qualified",
        "market_sides": json.dumps([{"flagged_side": "Baylor +3.5", "reaction": "Watch",
                                    "source_latest_at": "2026-11-27T00:30:00Z"}]),
    }

    def publish():
        with board.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)

    publish()
    research = tmp_path / "verified.jsonl"
    row = {"sport": "ncaab", "event_id": "two", "source": "verified fixture",
           "status": "verified", "provider_event_id": "provider-two", "score_a": 80, "score_b": 75}
    research.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with connect(tmp_path / "state") as db:
        assert ingest_board(db, board) == 1
        assert ingest_board(db, board) == 0
        data["supported_side"] = ""
        publish()
        assert ingest_board(db, board) == 1
        assert db.execute("SELECT count(*) FROM decisions").fetchone()[0] == 2
        assert import_outcomes(db, research) == 1
        assert import_outcomes(db, research) == 1
        assert db.execute("SELECT count(*) FROM outcomes").fetchone()[0] == 1


def test_historical_mlb_cancellation_needs_unambiguous_fixture():
    away, home = "Seattle Mariners", "Chicago Cubs"
    start = "2026-07-02T17:00:00Z"
    fixture = {"matchup": f"{away} @ {home}", "scheduled_at": start}
    event = {
        "game_pk": "1", "official_date": "2026-07-02", "scheduled_start": start,
        "status": "Cancelled", "state": "postponed",
        "away": {"aliases": [team_key(away)]}, "home": {"aliases": [team_key(home)]},
    }
    status, evidence = resolve_mlb(fixture, {"1": event})
    assert (status, evidence["provider_event_id"]) == ("void", "1")
    assert resolve_mlb(fixture, {"1": dict(event, status="Postponed")})[0] == "unresolved"
    assert resolve_mlb(fixture, {"1": event, "2": dict(event, game_pk="2")})[0] == "unresolved"


@pytest.mark.skipif(os.name != "posix", reason="Production compactor uses Linux flock.")
def test_compaction_preserves_exact_bytes_and_conflicting_source(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "detail_archive", Path(__file__).resolve().parents[1] / "ops" / "redfox-detail-archive.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    active = tmp_path / "data" / "anomaly_event_details"
    active.mkdir(parents=True)
    old = active / "nba--1.json"
    new = active / "nba--2.json"
    old.write_bytes(b'[{"odds":"-110","side":"one"}]\n')
    new.write_bytes(b"[]\n")
    old_time = time.time() - 9 * 86400
    os.utime(old, (old_time, old_time))
    original = old.read_bytes()
    public = tmp_path / "public"
    lock = tmp_path / "runner.lock"
    result = module.maintain(tmp_path, public, lock, days=7, limit=1000, retire=True)
    saved = public / "data" / "anomaly_event_details" / "nba--1.json.gz"
    assert result["retired"] == 1
    assert not old.exists()
    assert gzip.decompress(saved.read_bytes()) == original
    assert new.read_bytes() == b"[]\n"
    old.write_bytes(b"[2]\n")
    os.utime(old, (old_time, old_time))
    result = module.maintain(tmp_path, public, lock, days=7, limit=1000, retire=True)
    assert result["conflict"] == 1
    assert old.read_bytes() == b"[2]\n"
    assert gzip.decompress(saved.read_bytes()) == original
