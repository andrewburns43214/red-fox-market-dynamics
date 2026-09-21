"""Slow finals maintenance should work on recently completed games only."""

from pathlib import Path

import pandas as pd

import main


def test_scheduled_finals_lookup_skips_future_and_old_unresolved_games(tmp_path, monkeypatch):
    path = tmp_path / "snapshots.csv"
    pd.DataFrame([
        {"timestamp": "2026-08-20T16:00:00Z", "sport": "mlb", "game_id": "old", "game": "Old Away @ Old Home", "side": "Old Away", "dk_start_iso": "2026-08-20T20:00:00Z"},
        {"timestamp": "2026-09-20T16:00:00Z", "sport": "mlb", "game_id": "recent", "game": "Recent Away @ Recent Home", "side": "Recent Away", "dk_start_iso": "2026-09-20T20:00:00Z"},
        {"timestamp": "2026-09-21T16:00:00Z", "sport": "mlb", "game_id": "future", "game": "Future Away @ Future Home", "side": "Future Away", "dk_start_iso": "2026-09-22T20:00:00Z"},
    ]).to_csv(path, index=False)
    monkeypatch.setattr(main, "SNAPSHOT_CSV", str(path))
    calls = []
    monkeypatch.setattr(main, "get_espn_finals_map", lambda sport, games, dates: calls.append((sport, games, dates)) or {})
    monkeypatch.setattr(main, "update_final_scores_history", lambda: None)

    main.update_snapshots_with_espn_finals(lookback_days=14, now="2026-09-21T16:00:00Z")

    assert calls == [("mlb", ["Recent Away @ Recent Home"], ["20260920"])]


def test_scheduled_maintenance_gives_live_board_priority():
    root = Path(__file__).resolve().parents[1]
    source = (root / "run_maintenance.sh").read_text(encoding="utf-8")
    assert 'ionice -c3 nice -n 15 "$PY" main.py report_maintenance' in source
    assert "update_snapshots_with_espn_finals(lookback_days=14)" in (root / "main.py").read_text(encoding="utf-8")
