"""Serve byte-identical historic game details from compressed storage."""

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

NAME = re.compile(r"(?:nfl|ncaaf|nba|ncaab|nhl|mlb|ufc)--[A-Za-z0-9._%+-]+\.json$")
BLOCK_SIZE = 1024 * 1024


def checksum(handle):
    result = hashlib.sha256()
    for chunk in iter(lambda: handle.read(BLOCK_SIZE), b""):
        result.update(chunk)
    return result.hexdigest()


def metadata(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def archive_detail(source, destination, lock, *, retire=False, wait_seconds=0):
    original = metadata(source)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        if destination.exists():
            with gzip.open(destination, "rb") as stored, source.open("rb") as data:
                if checksum(stored) != checksum(data):
                    return "conflict"
        else:
            with source.open("rb") as data, temporary.open("wb") as output:
                expected = checksum(data)
                data.seek(0)
                with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0,
                                   compresslevel=3) as compressed:
                    shutil.copyfileobj(data, compressed, BLOCK_SIZE)
                output.flush()
                os.fsync(output.fileno())
            with gzip.open(temporary, "rb") as stored:
                if checksum(stored) != expected:
                    raise OSError(f"Archive readback did not match {source.name}")
            if metadata(source) != original:
                return "changed"
            temporary.chmod(0o644)
            os.replace(temporary, destination)
            fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        if not retire:
            return "archived"
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "busy"
                time.sleep(min(2, remaining))
        try:
            # The same lock guards scheduled publication. A changed path is kept.
            if metadata(source) != original:
                return "changed"
            source.unlink()
            return "retired"
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    finally:
        temporary.unlink(missing_ok=True)


def maintain(root, public, runner_lock, *, days, limit, retire=False, now=None, wait_seconds=0):
    origin = root / "data" / "anomaly_event_details"
    destination = public / "data" / "anomaly_event_details"
    destination.mkdir(parents=True, exist_ok=True, mode=0o755)
    cutoff = (now if now is not None else time.time()) - days * 86400
    result = {
        "scanned": 0, "archived": 0, "retired": 0, "changed": 0, "busy": 0,
        "conflict": 0, "reclaimed_bytes": 0, "processed_bytes": 0,
        "available_bytes": shutil.disk_usage(root).free,
    }
    with runner_lock.open("a") as lock:
        for source in sorted(origin.glob("*.json"), key=lambda p: p.stat().st_mtime):
            if not NAME.fullmatch(source.name) or source.is_symlink():
                continue
            st = source.stat()
            if st.st_mtime >= cutoff:
                break
            if result["processed_bytes"] and result["processed_bytes"] + st.st_size > limit:
                break
            result["scanned"] += 1
            result["processed_bytes"] += st.st_size
            outcome = archive_detail(
                source, destination / f"{source.name}.gz", lock,
                retire=retire, wait_seconds=wait_seconds if result["scanned"] == 1 else 0,
            )
            result[outcome] += 1
            if outcome == "retired":
                result["reclaimed_bytes"] += st.st_size
    result["available_bytes"] = shutil.disk_usage(root).free
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/opt/red-fox-market-dynamics"))
    parser.add_argument("--public", type=Path, default=Path("/var/lib/redfox-history/public"))
    parser.add_argument("--runner-lock", type=Path, default=Path("/var/lock/redfox_run.lock"))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--max-mib", type=int, default=192)
    parser.add_argument("--retire", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    summary = maintain(args.root, args.public, args.runner_lock, days=args.days,
                       limit=args.max_mib * 1024 * 1024, retire=args.retire, wait_seconds=args.wait_seconds)
    print(json.dumps(summary, sort_keys=True))
    if summary["available_bytes"] < 10 * 1024**3:
        print("WARNING historical storage has less than 10 GiB free", flush=True)


if __name__ == "__main__":
    main()
