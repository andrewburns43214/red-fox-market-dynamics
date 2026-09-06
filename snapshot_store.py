"""Cooperating snapshot writers: short shared transactions and atomic publication.

The lock belongs to a stable sidecar, never the CSV inode replaced on commit.
Network requests and final-score calculation must happen outside the transaction.
"""
from contextlib import contextmanager
import csv
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time

import pandas as pd

SCORE_FIELDS = ['final_score_for', 'final_score_against']


@contextmanager
def snapshot_lock(path, timeout=30):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / ('.' + path.name + '.lock')).open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b'\0')
                lock.flush()
            def acquire():
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            def release():
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            def acquire():
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            def release():
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        deadline = time.monotonic() + timeout
        while True:
            try:
                acquire()
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Snapshot writer lock timed out: {path}')
                time.sleep(.05)
            except OSError as error:
                # Windows reports lock contention as EACCES/EDEADLK.
                if os.name != 'nt' or error.errno not in (13, 36):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Snapshot writer lock timed out: {path}') from error
                time.sleep(.05)
        try:
            yield path
        finally:
            release()


@contextmanager
def atomic_snapshot(path, *, commit_if=None):
    """Caller holds snapshot_lock; failures before replace leave the CSV intact."""
    path = Path(path)
    existing = path.stat() if path.exists() else None
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        if commit_if is not None and not commit_if():
            return
        with temporary.open('r+b') as completed:
            if existing is not None and hasattr(os, 'fchown'):
                os.fchown(completed.fileno(), existing.st_uid, existing.st_gid)
            os.chmod(temporary, stat.S_IMODE(existing.st_mode) if existing else 0o644)
            completed.flush()
            os.fsync(completed.fileno())
        os.replace(temporary, path)
        if os.name != 'nt':
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def append_snapshot_rows(path, rows, fields):
    """Publish a complete capture batch; preserve existing history bytes/schema."""
    with snapshot_lock(path) as path:
        has_content = path.exists() and path.stat().st_size > 0
        header = list(fields)
        if has_content:
            with path.open(newline='', encoding='utf-8') as source:
                header = next(csv.reader(source))
            if len(header) != len(set(header)) or not set(fields).issubset(header):
                raise ValueError('Snapshot CSV header is incompatible; original file preserved')
        with atomic_snapshot(path) as temporary:
            if has_content:
                with path.open('rb') as source, temporary.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
                    source.seek(-1, os.SEEK_END)
                    if source.read(1) not in (b'\n', b'\r'):
                        destination.write(b'\n')
            with temporary.open('a', newline='', encoding='utf-8') as destination:
                writer = csv.DictWriter(destination, fieldnames=header)
                if not has_content:
                    writer.writeheader()
                writer.writerows(rows)


def merge_final_scores(path, evaluated, original_scores):
    """Compare-and-set sparse finals onto the latest CSV, never an old full copy.

Match every observed non-score column plus both original scores exactly. Appends,
duplicate historical rows, extra columns and concurrent finals are retained.
Conflicting stale proposals cannot overwrite a newer committed final-score pair.
"""
    if not evaluated.index.equals(original_scores.index):
        raise ValueError('Final-score proposal no longer aligns with observed rows')
    proposed = evaluated[SCORE_FIELDS].astype('string').fillna('').astype(str)
    observed = original_scores[SCORE_FIELDS].astype('string').fillna('').astype(str)
    changed = (proposed != observed).any(axis=1)
    identity = [column for column in evaluated.columns if column not in SCORE_FIELDS]
    if not {'timestamp', 'sport', 'game_id', 'game', 'side'}.issubset(identity):
        raise ValueError('Final-score proposal lacks exact snapshot identity columns')
    patches = {}
    for values, old, new in zip(evaluated.loc[changed, identity].itertuples(index=False, name=None),
                                observed.loc[changed].itertuples(index=False, name=None),
                                proposed.loc[changed].itertuples(index=False, name=None)):
        key = tuple(values) + old
        if key in patches and patches[key] != new:
            raise ValueError('Conflicting final-score proposals for identical observed rows')
        patches[key] = new
    with snapshot_lock(path) as path:
        with path.open(newline='', encoding='utf-8') as source:
            header = next(csv.reader(source))
        if len(header) != len(set(header)) or not set(identity).issubset(header):
            raise ValueError('Snapshot identity schema changed; original file preserved')
        source_width = len(header)
        schema_changed = any(column not in header for column in SCORE_FIELDS)
        header += [column for column in SCORE_FIELDS if column not in header]
        if not patches and not schema_changed:
            return 0
        applied = 0
        match_positions = [header.index(column) for column in identity + SCORE_FIELDS]
        score_positions = [header.index(column) for column in SCORE_FIELDS]
        # Stream under the lock: do not allocate/re-serialize a second full
        # dataframe while the capture writer waits. Close the read handle before
        # replace so this transaction also works on Windows.
        with atomic_snapshot(path, commit_if=lambda: applied or schema_changed) as temporary:
            with path.open(newline='', encoding='utf-8') as source, temporary.open('w', newline='', encoding='utf-8') as destination:
                reader = csv.reader(source)
                next(reader)
                writer = csv.writer(destination)
                writer.writerow(header)
                for row in reader:
                    if not row:
                        continue  # Match pandas' existing skip_blank_lines behavior.
                    if len(row) > source_width:
                        raise ValueError('Malformed snapshot row; original file preserved')
                    row.extend([''] * (len(header) - len(row)))
                    new = patches.get(tuple(row[position] for position in match_positions))
                    if new is not None:
                        for position, value in zip(score_positions, new):
                            row[position] = value
                        applied += 1
                    writer.writerow(row)
        return applied
