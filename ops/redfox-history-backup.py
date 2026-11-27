"""Back up irreplaceable Red Fox data to an operator-configured private restic repository."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess

DATA = (
    "snapshots.csv", "snapshot_archive", "_pre_v12_archive", "v2_backup_epoch",
    "snapshots_backup.csv", "snapshots_PREBACKFILL.csv", "snapshots.bak_before_movement.csv",
    "final_scores_history.csv", "ufc_results_history.csv", "red_fox_favorite_tracking.csv",
    "red_fox_favorite_freeze_candidates.csv", "red_fox_favorite_shadow.csv",
    "red_fox_favorite_review.csv", "performance_favorites.csv", "performance_ledger.csv",
    "favorite_performance.json", "clv_closing_lines.csv", "anomaly_action_ledger.csv",
    "open_registry.csv", "l1_sharp.csv", "l2_consensus.csv", "mlb_fixture_context",
)
CONFIGS = (
    "/etc/redfox-expected-revision", "/etc/redfox-ufc-integrity.json",
    "/etc/redfox-nhl-release.json", "/etc/redfox-nhl-official-since",
)


def snapshot_sqlite(source, target):
    if not source.is_file():
        return False
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True, timeout=20) as src:
        with sqlite3.connect(target, timeout=20) as dst:
            src.backup(dst, pages=128, sleep=0.1)
            if dst.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError(f"Invalid SQLite backup: {source.name}")
    return True


def backup_paths(root, state, stage, manual, configs):
    data = root / "data"
    paths = [data / f for f in DATA if (data / f).exists()]
    if (state / "research").exists():
        paths.append(state / "research")
    paths.extend(sorted(data.glob("snapshots.csv.bak_*")))
    paths.extend(p for p in configs if p.exists())
    # The index is rebuildable except for observed decisions and reconciliations.
    # Export those private tables with a single consistent read transaction.
    source = state / "history.sqlite3"
    if source.is_file():
        with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True, timeout=20) as db:
            db.execute("BEGIN")
            for table in ("decisions", "outcomes", "unresolved"):
                target = stage / f"{table}.jsonl"
                with target.open("w", encoding="utf-8") as stream:
                    cursor = db.execute(f"SELECT * FROM {table} ORDER BY sport,event_id")
                    keys = [column[0] for column in cursor.description]
                    for row in cursor:
                        stream.write(json.dumps(dict(zip(keys, row)), ensure_ascii=False, separators=(",", ":")) + "\n")
                paths.append(target)
    manual_copy = stage / "manual-decisions.sqlite3"
    if snapshot_sqlite(manual, manual_copy):
        paths.append(manual_copy)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("backup", "check"))
    parser.add_argument("--root", type=Path, default=Path("/opt/red-fox-market-dynamics"))
    parser.add_argument("--state", type=Path, default=Path("/var/lib/redfox-history/private"))
    parser.add_argument("--manual", type=Path, default=Path("/var/lib/redfox-manual-favorites/decisions.sqlite3"))
    args = parser.parse_args()
    repository = os.environ.get("RESTIC_REPOSITORY", "")
    if not repository.startswith(("s3:", "sftp:", "rest:", "rclone:", "azure:", "gs:", "b2:", "swift:")):
        raise SystemExit("A private off-host restic repository must be configured before backup can run.")
    if shutil.which("restic") is None:
        raise SystemExit("restic is unavailable.")
    if args.command == "check":
        subprocess.run(["restic", "check", "--read-data-subset=5%"], check=True)
        return
    # Stable internal file paths allow restic to reuse metadata and chunks.
    stage = args.state / "backup-stage"
    stage.mkdir(mode=0o700, parents=True, exist_ok=True)
    paths = backup_paths(args.root, args.state, stage, args.manual, map(Path, CONFIGS))
    if not paths:
        raise SystemExit("No Red Fox backup sources were found.")
    try:
        subprocess.run(["restic", "backup", "--tag", "redfox-history", *map(str, paths)], check=True)
    finally:
        for name in ("decisions.jsonl", "outcomes.jsonl", "unresolved.jsonl", "manual-decisions.sqlite3"):
            (stage / name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
