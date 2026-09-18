"""Temporary, aggregate-only MLB publication probe for the live runner."""

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TARGET = ROOT / "site" / "mlb-status-20260918.json"


def latest_run(db, kind, sport=""):
    row = db.execute(
        "SELECT state, started_at, finished_at, detail FROM runs "
        "WHERE kind = ? AND sport = ? ORDER BY started_at DESC LIMIT 1",
        (kind, sport),
    ).fetchone()
    return dict(zip(("state", "started_at", "finished_at", "detail"), row)) if row else None


def main():
    result = {"as_of": datetime.now(timezone.utc).isoformat()}
    try:
        with sqlite3.connect(f"file:{(DATA / 'publication_coverage.sqlite3').as_posix()}?mode=ro", uri=True) as db:
            result["scrape"] = latest_run(db, "SCRAPE", "mlb")
            result["capture"] = latest_run(db, "CAPTURE", "mlb")
            result["publication"] = latest_run(db, "PUBLICATION")
    except (OSError, sqlite3.Error) as error:
        result["coverage_error"] = type(error).__name__

    try:
        with (DATA / "publication_coverage.csv").open(newline="", encoding="utf-8") as source:
            rows = [row for row in csv.DictReader(source) if row.get("sport") == "mlb"]
        result["mlb_markets"] = len(rows)
        result["states"] = dict(Counter(row.get("state", "") for row in rows))
        result["latest_capture_at"] = max((row.get("last_capture_at", "") for row in rows), default="")
        result["latest_pair_at"] = max((row.get("last_complete_pair_at", "") for row in rows), default="")
        result["coverage_as_of"] = max((row.get("as_of", "") for row in rows), default="")
    except (OSError, csv.Error) as error:
        result["market_error"] = type(error).__name__

    try:
        with (DATA / "anomaly_board.csv").open(newline="", encoding="utf-8") as source:
            result["mlb_board_rows"] = sum(row.get("sport") == "mlb" for row in csv.DictReader(source))
    except (OSError, csv.Error) as error:
        result["board_error"] = type(error).__name__

    try:
        freshness = json.loads((DATA / "freshness.json").read_text(encoding="utf-8"))
        result["engine_ts"] = freshness.get("engine_ts")
        result["dk_ts"] = freshness.get("dk_ts")
    except (OSError, ValueError) as error:
        result["freshness_error"] = type(error).__name__

    temporary = TARGET.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    temporary.replace(TARGET)


if __name__ == "__main__":
    main()
