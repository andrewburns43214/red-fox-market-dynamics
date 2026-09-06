"""Read-only coverage health check; output includes in-window football counts."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone


def check(data_dir, now=None, max_age_minutes=25):
    root = Path(data_dir)
    now = now or datetime.now(timezone.utc)
    issues = []
    try:
        summary = json.loads((root / "publication_coverage.json").read_text(encoding="utf-8"))
        age = (now - datetime.fromisoformat(summary["as_of"])).total_seconds() / 60
        if age < -1 or age > max_age_minutes:
            issues.append("COVERAGE_REPORT_STALE_OR_FUTURE")
        if hashlib.sha256((root / "anomaly_board.csv").read_bytes()).hexdigest() != summary["board_sha256"]:
            issues.append("COVERAGE_EXPORT_VERSION_MISMATCH")
        if not summary.get("source_census_complete"):
            issues.append("SOURCE_CENSUS_INCOMPLETE")
        if summary.get("unexplained_gaps") or summary.get("publication_conflicts"):
            issues.append("PUBLICATION_RECONCILIATION_FAILED")
        for sport, scrape in summary.get("latest_scrapes", {}).items():
            if sport not in summary.get("expected_active_sports", []):
                continue
            scrape_age = (now - datetime.fromisoformat(scrape["started_at"])).total_seconds() / 60
            if scrape_age > max_age_minutes:
                issues.append(f"{sport}:SOURCE_CAPTURE_STALE")
        for sport, stats in summary["sports"].items():
            if sport == "ALL":
                continue
            if stats["eligible_in_window"] != stats["published"] + sum(stats["excluded_by_reason"].values()):
                issues.append(f"{sport}:UNACCOUNTED_MARKET")
            if stats["excluded_by_reason"].get("STALE_CAPTURE", 0):
                issues.append(f"{sport}:STALE_IN_WINDOW_MARKETS")
        with sqlite3.connect((root / "publication_coverage.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
            latest = db.execute("SELECT state,started_at FROM runs WHERE kind='PUBLICATION' ORDER BY started_at DESC LIMIT 1").fetchone()
            if latest and latest[0] == "PUBLICATION_FAILED":
                issues.append("LATEST_PUBLICATION_FAILED")
            captures = db.execute("SELECT sport,state FROM runs WHERE kind='CAPTURE' ORDER BY started_at").fetchall()
            if any(state == "CAPTURE_FAILED" for state in dict(captures).values()):
                issues.append("LATEST_CAPTURE_FAILED")
            running = db.execute("SELECT kind,started_at FROM runs WHERE state='RUNNING'").fetchall()
            if any((now - datetime.fromisoformat(start)).total_seconds() > max_age_minutes * 60 for _, start in running):
                issues.append("INTERRUPTED_COVERAGE_RUN")
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
        return {"issues": ["COVERAGE_EVIDENCE_UNAVAILABLE"], "detail": str(error)}, False
    summary["issues"] = issues
    return summary, not issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-age-minutes", type=float, default=25)
    args = parser.parse_args()
    result, healthy = check(args.data_dir, max_age_minutes=args.max_age_minutes)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if healthy else 1)


if __name__ == "__main__":
    main()
