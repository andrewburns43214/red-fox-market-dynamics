import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
from pathlib import Path
import threading

import pandas as pd
import pytest

import main
import snapshot_store as store


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def pair(game_id='old', timestamp='2026-09-01T12:00:00+00:00'):
    return [dict(timestamp=timestamp, sport='ncaaf', game_id=game_id,
                 game='UNLV @ Hawaii', side=side, market='splits', bets_pct='50', money_pct='50',
                 open_line=line, current_line=line, injury_news='', key_number_note='',
                 dk_start_iso=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat())
            for side, line in [('UNLV', 'UNLV @ -110'), ('Hawaii', 'Hawaii @ -110')]]


def proposal(path):
    frame = read(path)
    for col in store.SCORE_FIELDS:
        if col not in frame:
            frame[col] = ''
    observed = frame[store.SCORE_FIELDS].copy()
    frame[store.SCORE_FIELDS] = ['21', '17']
    return frame, observed


def writer_process(path, operation, ready, release, attempting, done):
    """Real independent OS writer; optionally pause with its lock before replace."""
    import snapshot_store as child
    original_replace = child.os.replace
    if ready is not None:
        def paused_replace(source, target):
            ready.set()
            if not release.wait(15):
                raise RuntimeError('Parent did not release publication barrier')
            original_replace(source, target)
        child.os.replace = paused_replace
    if operation == 'maintenance':
        frame, observed = proposal(path)
    attempting.set()
    if operation == 'append':
        child.append_snapshot_rows(path, pair('new'), main.SNAPSHOT_FIELDS)
    else:
        child.merge_final_scores(path, frame, observed)
    done.set()


def test_safe_reproduction_of_old_read_append_rewrite_loses_new_pair(tmp_path):
    path = tmp_path / 'snapshots.csv'
    pd.DataFrame(pair()).to_csv(path, index=False)
    stale_maintenance_copy = read(path)
    store.append_snapshot_rows(path, pair('new'), main.SNAPSHOT_FIELDS)
    assert len(read(path)) == 4
    # This is the exact former lost-update sequence, confined to test data.
    stale_maintenance_copy.to_csv(path, index=False)
    assert len(read(path)) == 2
    assert 'new' not in set(read(path).game_id)


def test_real_maintenance_preserves_overlapping_capture_and_publisher_sees_it(tmp_path, monkeypatch):
    import refresh_anomaly_board as refresh
    from publication_coverage import PublicationCoverage

    path = tmp_path / 'snapshots.csv'
    old = pair(timestamp=(datetime.now(timezone.utc) - timedelta(minutes=4)).isoformat())
    # Duplicate historical observations and an unknown historical column survive.
    historical = pd.DataFrame(old + old)
    historical['forensic_note'] = ['comma, note', 'Hawaiʻi\nline two', '', 'duplicate']
    historical.to_csv(path, index=False)
    monkeypatch.setattr(main, 'SNAPSHOT_CSV', str(path))
    monkeypatch.setattr(main, 'update_final_scores_history', lambda: None)
    fetched = threading.Event()
    release = threading.Event()
    def finals(*args, **kwargs):
        fetched.set()
        assert release.wait(10)
        return {'UNLV @ Hawaii': (21, 17)}
    monkeypatch.setattr(main, 'get_espn_finals_map', finals)
    with ThreadPoolExecutor(max_workers=2) as pool:
        maintenance = pool.submit(main.update_snapshots_with_espn_finals)
        try:
            assert fetched.wait(5)
            capture = pool.submit(main.append_snapshot, pair(), 'ncaaf')
            capture.result(timeout=5)  # ESPN work must not block capture storage.
            after_capture = read(path)
        finally:
            release.set()
        maintenance.result(timeout=10)
    after = read(path)
    assert len(after) == 6
    pd.testing.assert_frame_equal(after[historical.columns], after_capture[historical.columns])
    pd.testing.assert_frame_equal(after.iloc[:4][historical.columns], historical)
    assert after.iloc[:4].final_score_for.tolist() == ['21', '17', '21', '17']
    assert after.iloc[4:].final_score_for.tolist() == ['', '']

    monkeypatch.setattr(refresh, 'DATA', tmp_path)
    monkeypatch.setattr(refresh, 'update_action_ledger', lambda *args: 0)
    monkeypatch.setattr(refresh, 'apply_recorded_signals', lambda frame, *args: frame)
    monkeypatch.setattr(refresh, 'rebuild_action_results', lambda *args: 0)
    refresh._refresh(PublicationCoverage(tmp_path, pd.Timestamp.now(tz='UTC')))
    board = read(tmp_path / 'anomaly_board.csv')
    assert board.game_id.tolist() == ['old']
    coverage = read(tmp_path / 'publication_coverage.csv')
    assert coverage.state.tolist() == ['PUBLISHED']
    assert pd.Timestamp(coverage.iloc[0].last_complete_pair_at) == pd.Timestamp(after.timestamp.max())
    assert float(coverage.iloc[0].complete_pair_age_minutes) < 1
    assert len(json.loads(board.iloc[0].market_sides)) == 2


@pytest.mark.parametrize('first', ['maintenance', 'append'])
def test_two_processes_serialize_commit_without_losing_a_pair(tmp_path, first):
    path = tmp_path / 'snapshots.csv'
    pd.DataFrame(pair()).to_csv(path, index=False)
    ctx = multiprocessing.get_context('spawn')
    ready, release = ctx.Event(), ctx.Event()
    attempt_a, attempt_b, done_a, done_b = [ctx.Event() for _ in range(4)]
    a = ctx.Process(target=writer_process, args=(path, first, ready, release, attempt_a, done_a))
    second = 'append' if first == 'maintenance' else 'maintenance'
    b = ctx.Process(target=writer_process, args=(path, second, None, release, attempt_b, done_b))
    try:
        a.start()
        assert ready.wait(10)
        b.start()
        assert attempt_b.wait(10)
        assert not done_b.wait(.2)
        assert len(read(path)) == 2  # Reader still sees the complete old version.
        release.set()
        a.join(15); b.join(15)
        assert a.exitcode == b.exitcode == 0
        final = read(path)
        assert final.game_id.tolist() == ['old', 'old', 'new', 'new']
        assert final.iloc[:2].final_score_for.tolist() == ['21', '21']
    finally:
        release.set()
        for process in (a, b):
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(5)


@pytest.mark.parametrize('operation', ['append', 'maintenance'])
def test_failed_atomic_commit_preserves_history_and_releases_lock(tmp_path, monkeypatch, operation):
    path = tmp_path / 'snapshots.csv'
    pd.DataFrame(pair()).to_csv(path, index=False)
    original = path.read_bytes()
    frame, observed = proposal(path)
    def fail(*args):
        raise OSError('injected replace failure')
    with monkeypatch.context() as patch:
        patch.setattr(store.os, 'replace', fail)
        with pytest.raises(OSError, match='injected'):
            if operation == 'append':
                store.append_snapshot_rows(path, pair('new'), main.SNAPSHOT_FIELDS)
            else:
                store.merge_final_scores(path, frame, observed)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.tmp'))
    store.append_snapshot_rows(path, pair('recovery'), main.SNAPSHOT_FIELDS)
    assert len(read(path)) == 4


def test_killed_writer_cannot_truncate_history_or_leave_lock_held(tmp_path):
    path = tmp_path / 'snapshots.csv'
    pd.DataFrame(pair()).to_csv(path, index=False)
    original = path.read_bytes()
    ctx = multiprocessing.get_context('spawn')
    ready, release, attempting, done = [ctx.Event() for _ in range(4)]
    worker = ctx.Process(target=writer_process, args=(path, 'append', ready, release, attempting, done))
    try:
        worker.start()
        assert ready.wait(10)
        worker.terminate(); worker.join(10)
        assert worker.exitcode is not None
        assert path.read_bytes() == original
        store.append_snapshot_rows(path, pair('recovery'), main.SNAPSHOT_FIELDS)
        assert len(read(path)) == 4
    finally:
        if worker.is_alive():
            worker.terminate(); worker.join(5)


def test_stale_finals_proposal_cannot_overwrite_newer_finals_or_other_columns(tmp_path):
    path = tmp_path / 'snapshots.csv'
    pd.DataFrame(pair() + pair()).to_csv(path, index=False)
    stale, observed = proposal(path)
    fresh = stale.copy(); fresh[store.SCORE_FIELDS] = ['24', '20']
    store.merge_final_scores(path, fresh, observed)
    newer = read(path)
    newer['extra_metadata'] = ['a', 'b', 'c', 'd']
    newer.to_csv(path, index=False)  # Simulate another completed schema extension.
    assert store.merge_final_scores(path, stale, observed) == 0
    pd.testing.assert_frame_equal(read(path), newer)


def test_append_preserves_extended_reordered_schema_and_history_bytes(tmp_path):
    path = tmp_path / 'snapshots.csv'
    existing = pd.DataFrame(pair())
    existing['final_score_for'] = ['21', '17']
    existing['final_score_against'] = ['17', '21']
    existing = existing[list(reversed(existing.columns))]
    existing.to_csv(path, index=False)
    original = path.read_bytes()
    store.append_snapshot_rows(path, pair('new'), main.SNAPSHOT_FIELDS)
    assert path.read_bytes().startswith(original)
    final = read(path)
    pd.testing.assert_frame_equal(final.iloc[:2], existing)
    assert final.iloc[2:].final_score_for.tolist() == ['', '']
    assert final.iloc[2:].game_id.tolist() == ['new', 'new']


def test_incompatible_header_fails_without_touching_history(tmp_path):
    path = tmp_path / 'snapshots.csv'
    path.write_text('sport,game_id\nncaaf,old\n')
    original = path.read_bytes()
    with pytest.raises(ValueError, match='header'):
        store.append_snapshot_rows(path, pair('new'), main.SNAPSHOT_FIELDS)
    assert path.read_bytes() == original


def test_lock_timeout_is_explicit_and_does_not_bypass_owner(tmp_path):
    path = tmp_path / 'snapshots.csv'
    with store.snapshot_lock(path):
        with pytest.raises(TimeoutError):
            with store.snapshot_lock(path, timeout=.05):
                pytest.fail('Contending writer bypassed the lock')


def test_two_processes_create_one_header_and_keep_both_capture_batches(tmp_path):
    path = tmp_path / 'snapshots.csv'
    ctx = multiprocessing.get_context('spawn')
    ready, release = ctx.Event(), ctx.Event()
    attempt_a, attempt_b, done_a, done_b = [ctx.Event() for _ in range(4)]
    a = ctx.Process(target=writer_process, args=(path, 'append', ready, release, attempt_a, done_a))
    b = ctx.Process(target=writer_process, args=(path, 'append', None, release, attempt_b, done_b))
    try:
        a.start(); assert ready.wait(10)
        b.start(); assert attempt_b.wait(10)
        assert not done_b.wait(.2)
        assert not path.exists()  # No header-only or half-pair publication.
        release.set()
        a.join(15); b.join(15)
        assert a.exitcode == b.exitcode == 0
        final = read(path)
        assert len(final) == 4
        assert final.side.tolist() == ['UNLV', 'Hawaii', 'UNLV', 'Hawaii']
        assert list(final.columns) == main.SNAPSHOT_FIELDS
    finally:
        release.set()
        for process in (a, b):
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(5)


def test_conflicting_duplicate_proposals_fail_without_modifying_history(tmp_path):
    path = tmp_path / 'snapshots.csv'
    rows = pair()
    pd.DataFrame([rows[0], rows[0]]).to_csv(path, index=False)
    original = path.read_bytes()
    evaluated, observed = proposal(path)
    evaluated.loc[1, 'final_score_for'] = '99'
    with pytest.raises(ValueError, match='Conflicting'):
        store.merge_final_scores(path, evaluated, observed)
    assert path.read_bytes() == original


def test_incomplete_identity_cannot_apply_finals_to_unrelated_rows(tmp_path):
    path = tmp_path / 'snapshots.csv'
    pd.DataFrame(pair()).to_csv(path, index=False)
    original = path.read_bytes()
    evaluated, observed = proposal(path)
    with pytest.raises(ValueError, match='identity'):
        store.merge_final_scores(path, evaluated.drop(columns=['timestamp']), observed)
    assert path.read_bytes() == original
