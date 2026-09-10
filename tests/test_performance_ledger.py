import json
from pathlib import Path

import pandas as pd

from performance_ledger import LEDGER_COLUMNS, _utc_series, update_performance_ledger


def frozen_row(**updates):
    side = {
        "flagged_side": "NY Mets",
        "current_line": "-105",
        "reaction": "Contrarian",
        "anomaly_chips": "Contrarian | One-Way",
    }
    row = {
        "sport": "mlb", "game_id": "101", "game": "NY Mets @ MIA Marlins",
        "kickoff_iso": "2026-09-08T17:10:00Z", "frozen_at_utc": "2026-09-08T17:10:05Z",
        "state_as_of_utc": "2026-09-08T17:09:00Z",
        "final_pregame_state_at_utc": "2026-09-08T17:09:00Z",
        "freeze_method": "published_state_latest_at_or_before_start",
        "market_display": "MONEYLINE", "flagged_side": "NY Mets",
        "market_sides": json.dumps([side, {"flagged_side": "MIA Marlins", "current_line": "-115"}]),
        "supported_side": "NY Mets", "red_fox_favorite": "true", "favorite_side": "NY Mets",
        "favorite_rule_version": "red_fox_favorite_v2",
        "favorite_first_qualified_at": "2026-09-08T15:00:00Z",
        "favorite_final_qualified_at": "2026-09-08T17:09:00Z",
        "favorite_snapshot_id": "snapshot-101",
    }
    row.update(updates)
    return row


def write(frame, path: Path):
    frame.to_csv(path, index=False)


def test_ingests_directional_final_state_and_grades_without_reclassification(tmp_path):
    descriptive = frozen_row(
        game_id="102", game="A @ B", supported_side="", red_fox_favorite="false",
        favorite_side="", market_sides=json.dumps([{"flagged_side": "A", "current_line": "+110"}]),
    )
    fell_off = frozen_row(
        game_id="103", game="Team A @ Team B", market_display="SPREAD",
        supported_side="Team A +3.5", red_fox_favorite="false", favorite_side="",
        market_sides=json.dumps([
            {"flagged_side": "Team A +3.5", "current_line": "+3.5 (-110)", "reaction": "Follow", "anomaly_chips": "Follow | One-Way"},
            {"flagged_side": "Team B -3.5", "current_line": "-3.5 (-110)"},
        ]),
    )
    write(pd.DataFrame([frozen_row(), descriptive, fell_off]), tmp_path / "live_recent.csv")
    write(pd.DataFrame([
        {"recorded_at": "2026-09-08T14:00:00Z", "sport": "mlb", "game_id": "103", "market_display": "SPREAD", "favorite_state": "qualified"},
        {"recorded_at": "2026-09-08T16:00:00Z", "sport": "mlb", "game_id": "103", "market_display": "SPREAD", "favorite_state": "not_qualified"},
    ]), tmp_path / "red_fox_favorite_tracking.csv")
    write(pd.DataFrame([
        {"game_id": "101", "team1": "NY Mets", "team1_score": "7", "team2": "MIA Marlins", "team2_score": "5", "resolved_at_utc": "2026-09-08T21:00:00Z"},
        {"game_id": "103", "team1": "Team A", "team1_score": "20", "team2": "Team B", "team2_score": "24", "resolved_at_utc": "2026-09-08T22:00:00Z"},
    ]), tmp_path / "final_scores_history.csv")

    result = update_performance_ledger(tmp_path)
    assert result["rows"] == 2
    assert result["graded"] == 2
    assert result["supported"] == {"wins": 1, "losses": 1, "pushes": 0, "total_graded": 2, "win_rate_excluding_pushes": 0.5}
    assert result["favorites"]["wins"] == 1

    ledger = pd.read_csv(tmp_path / "performance_ledger.csv", dtype=str, keep_default_na=False)
    assert list(ledger.columns) == LEDGER_COLUMNS
    assert set(ledger["event_id"]) == {"101", "103"}
    mets = ledger[ledger.event_id.eq("101")].iloc[0]
    assert mets["final_pregame_price"] == "-105"
    assert mets["market_read"] == "Contrarian"
    assert mets["favorite_qualified"] == "yes"
    assert mets["grade"] == "W"
    spread = ledger[ledger.event_id.eq("103")].iloc[0]
    assert spread["final_pregame_line"] == "+3.5"
    assert spread["favorite_qualified"] == "no"
    assert spread["favorite_fell_off_before_kickoff"] == "yes"
    assert spread["grade"] == "L"

    frozen = pd.read_csv(tmp_path / "live_recent.csv", dtype=str, keep_default_na=False)
    frozen.loc[frozen.game_id.eq("101"), "supported_side"] = "MIA Marlins"
    write(frozen, tmp_path / "live_recent.csv")
    rerun = update_performance_ledger(tmp_path)
    unchanged = pd.read_csv(tmp_path / "performance_ledger.csv", dtype=str, keep_default_na=False)
    assert rerun["new_rows"] == 0
    assert unchanged.loc[unchanged.event_id.eq("101"), "supported_side"].iloc[0] == "NY Mets"


def test_v1_favorite_is_supported_side_only_and_unresolved_rows_remain_ungraded(tmp_path):
    write(pd.DataFrame([frozen_row(favorite_rule_version="red_fox_favorite_v1")]), tmp_path / "live_recent.csv")
    result = update_performance_ledger(tmp_path)
    assert result["rows"] == 1
    ledger = pd.read_csv(tmp_path / "performance_ledger.csv", dtype=str, keep_default_na=False)
    assert ledger.iloc[0]["favorite_qualified"] == "no"
    assert ledger.iloc[0]["grade"] == ""
    assert ledger.iloc[0]["audit_status"] == "awaiting_final_score"
    favorites = pd.read_csv(tmp_path / "performance_favorites.csv", dtype=str, keep_default_na=False)
    assert favorites.empty


def test_freeze_uses_latest_source_state_before_kickoff_and_rejects_post_start(tmp_path):
    earlier = frozen_row(supported_side="NY Mets", final_pregame_price="", final_pregame_state_at_utc="2026-09-08T17:08:00Z")
    latest = frozen_row(
        supported_side="MIA Marlins", favorite_side="MIA Marlins",
        final_pregame_state_at_utc="2026-09-08T17:09:59Z",
        market_sides=json.dumps([
            {"flagged_side": "NY Mets", "current_line": "-105"},
            {"flagged_side": "MIA Marlins", "current_line": "-115", "reaction": "Follow"},
        ]),
    )
    after = frozen_row(
        supported_side="NY Mets", final_pregame_state_at_utc="2026-09-08T17:10:01Z",
        frozen_at_utc="2026-09-08T17:12:00Z",
    )
    write(pd.DataFrame([earlier, after, latest]), tmp_path / "live_recent.csv")
    result = update_performance_ledger(tmp_path, attach_results=False)
    assert result["rows"] == 1
    ledger = pd.read_csv(tmp_path / "performance_ledger.csv", dtype=str, keep_default_na=False)
    assert ledger.iloc[0]["supported_side"] == "MIA Marlins"
    assert ledger.iloc[0]["final_pregame_state_at_utc"] == "2026-09-08T17:09:59+00:00"


def test_missing_source_state_is_not_an_official_freeze(tmp_path):
    write(pd.DataFrame([
        frozen_row(final_pregame_state_at_utc=""),
        frozen_row(game_id="104", final_pregame_state_at_utc=""),
    ]), tmp_path / "live_recent.csv")
    assert update_performance_ledger(tmp_path, attach_results=False)["rows"] == 0


def test_all_missing_source_times_remain_explicitly_utc_aware():
    parsed = _utc_series(pd.Series(["", ""]), pd.RangeIndex(2))
    assert isinstance(parsed.dtype, pd.DatetimeTZDtype)
    assert str(parsed.dtype) == "datetime64[ns, UTC]"


def test_admin_exports_are_protected_and_engine_refresh_does_not_import_ledger():
    root = Path(__file__).resolve().parents[1]
    nginx = (root / "deploy" / "nginx-redfox.production.conf").read_text(encoding="utf-8")
    access = (root / "access_verifier.py").read_text(encoding="utf-8")
    admin = (root / "site" / "admin-users.html").read_text(encoding="utf-8")
    refresh = (root / "refresh_anomaly_board.py").read_text(encoding="utf-8")
    runner = (root / "run_all_sports.sh").read_text(encoding="utf-8")
    service = (root / "deploy" / "redfox-performance.service").read_text(encoding="utf-8")
    timer = (root / "deploy" / "redfox-performance.timer").read_text(encoding="utf-8")
    assert "location = /_internal/admin-access" in nginx
    assert nginx.count("auth_request /_internal/admin-access;") == 3
    assert "/verify-admin" in access and "/functions/v1/admin-users" in access
    assert "bearer_token(self.headers.get(\"Authorization\"))" in access
    assert nginx.count("set $redfox_admin_authorization $http_authorization;") == 3
    assert "proxy_set_header Authorization $redfox_admin_authorization;" in nginx
    assert "Download Supported Side CSV" in admin
    assert "Download Red Fox Favorite CSV" in admin
    assert "document.cookie='redfox_access_token='+encodeURIComponent(session.access_token)" in admin
    assert "fetch(link.href,{credentials:'same-origin',headers:{'Authorization':'Bearer '+session.access_token}})" in admin
    assert "performance_ledger" not in refresh
    assert "performance_ledger" not in runner
    assert "performance_ledger.py --freeze-only" in service
    assert "OnActiveSec=15s" in timer
    assert "OnUnitActiveSec=2min" in timer
    assert "Persistent=true" not in timer
