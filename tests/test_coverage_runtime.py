import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from game_identity import clean_name, game_identity, match_games, team_identity
from main import infer_market_type, normalize_side_key, validate_snapshot_rows
from publication_coverage import CoverageStore, PublicationCoverage, ScrapeCoverage, keys
from refresh_anomaly_board import filter_publication_eligible_markets, latest_synchronized_market_rows
from coverage_monitor import check

NOW = pd.Timestamp("2026-09-05T18:00:00Z")
KICK = "2026-09-06T02:00:00Z"


@pytest.mark.parametrize("favorite,price,opponent,opponent_price", [
    ("South Carolina", "-100000", "Kent State", "+5000"),
    ("Utah", "-100000", "Idaho", "+5000"),
    ("Delaware", "-50000", "Merrimack", "+4000"),
    ("Iowa", "-100000", "Northern Illinois", "+5000"),
])
def test_exact_extreme_moneyline_patterns_preserve_both_sides(favorite, price, opponent, opponent_price):
    rows = []
    for side, odds in [(favorite, price), (opponent, opponent_price)]:
        market = infer_market_type(side, f"{side} @ {odds}")
        assert market == "MONEYLINE"
        rows.append(dict(sport="ncaaf", game_id="1", market_display=market, side=side,
                         side_key=normalize_side_key("ncaaf", market, side), timestamp=NOW))
    assert len(latest_synchronized_market_rows(pd.DataFrame(rows))) == 2
    assert latest_synchronized_market_rows(pd.DataFrame(rows[:1])).empty


@pytest.mark.parametrize("odds", ["0", "-99", "+99999999999", "-100000oops", "-100000.1"])
def test_extreme_price_fix_does_not_accept_malformed_odds(odds):
    assert infer_market_type("Utah", f"Utah @ {odds}") == ""


@pytest.mark.parametrize("variant", ["Hawaii", "Hawai'i", "Hawai‘i", "Hawaiʻi", "HAW", "Hawaii Rainbow Warriors"])
def test_hawaii_variants_have_one_identity(variant):
    assert team_identity(variant, "ncaaf") == ("hawaii", "IDENTIFIED")


@pytest.mark.parametrize("left,right,sport", [
    ("Nevada-Las Vegas", "UNLV Rebels", "ncaaf"),
    ("Ohio St.", "Ohio State", "ncaaf"),
    ("LSU", "Louisiana State", "ncaaf"),
    ("UConn", "Connecticut", "ncaaf"),
    ("W. Mich", "Western Michigan", "ncaaf"),
    ("N. Illinois", "Northern Illinois", "ncaaf"),
    ("Miami (FL)", "Miami Hurricanes", "ncaaf"),
    ("Miami (OH)", "Miami RedHawks", "ncaaf"),
    ("KC Chiefs", "Kansas City Chiefs", "nfl"),
    ("LV Raiders", "Las Vegas Raiders", "nfl"),
    ("NY Jets", "New York Jets", "nfl"),
])
def test_explicit_aliases(left, right, sport):
    assert team_identity(left, sport) == team_identity(right, sport)


@pytest.mark.parametrize("a,b", [("Miami (FL)", "Miami (OH)"), ("Nevada", "UNLV"),
                                   ("Northern Illinois", "Southern Illinois"), ("North Carolina", "North Carolina State"),
                                   ("Michigan", "Michigan State")])
def test_distinct_schools_never_merge(a, b):
    assert team_identity(a, "ncaaf")[0] != team_identity(b, "ncaaf")[0]


def event(eid="1", away="UNLV Rebels", home="Hawai‘i Rainbow Warriors"):
    return dict(id=eid, date=KICK, competitions=[dict(competitors=[
        dict(homeAway="away", team=dict(displayName=away, shortDisplayName=away)),
        dict(homeAway="home", team=dict(displayName=home, shortDisplayName=home))])])


def test_exact_name_matching_and_ambiguity_have_explicit_states():
    games = ["Nevada Las Vegas @ Hawaii", "Nevada @ Hawaii", "Miami @ Hawaii"]
    result = match_games(games, [event()], "ncaaf")
    assert result[games[0]] == KICK
    assert result.states[games[1]] == "ESPN_UNMATCHED"
    assert result.states[games[2]] == "AMBIGUOUS_TEAM_IDENTITY"
    collision = match_games(games[:1], [event("1"), event("2")], "ncaaf")
    assert collision.states[games[0]] == "ESPN_AMBIGUOUS"
    assert not collision[games[0]]


def test_shared_tokens_cannot_match_the_wrong_directional_school():
    result = match_games(["Northern Illinois @ Ohio State"],
                         [event(away="Southern Illinois", home="Ohio State")], "ncaaf")
    assert result.states["Northern Illinois @ Ohio State"] == "ESPN_UNMATCHED"


def dk_rows(game="UNLV @ Hawaii"):
    return [dict(sport="ncaaf", game_id="123", game=game, side=f"{side} 50.5",
                 current=f"{side} 50.5 @ -110", bets_pct=50, money_pct=50,
                 dk_start_iso=KICK, _source_league_verified=True) for side in ["Over", "Under"]]


def test_espn_outage_retains_valid_dk_but_ambiguous_names_are_quarantined(monkeypatch):
    monkeypatch.setattr("main.get_espn_kickoff_map", lambda *args: {})
    valid = dk_rows()
    accepted, _ = validate_snapshot_rows(valid, "ncaaf")
    assert len(accepted) == 2
    assert all(r["_validation_state"] == "ESPN_UNMATCHED" for r in accepted)
    ambiguous = dk_rows("Miami @ Hawaii")
    accepted, _ = validate_snapshot_rows(ambiguous, "ncaaf")
    assert not accepted
    assert all(r["_capture_exclusion_reason"] == "AMBIGUOUS_TEAM_IDENTITY" for r in ambiguous)
    invalid = dk_rows(); invalid[0]["game_id"] = ""
    assert len(validate_snapshot_rows(invalid, "ncaaf")[0]) == 1
    assert invalid[0]["_capture_exclusion_reason"] == "UNRESOLVED_EVENT_IDENTITY"


def header(gid="123", when="9/5, 10:00PM"):
    return f'''<select name="tb_eg"><option selected value="NCAA Football">CFB</option></select>
    <div class="tb-se"><div class="tb-se-title"><a href="/event/{gid}">UNLV @ Hawai'i</a><span>{when}</span></div>
    <div class="tb-se-head"><div>Total</div><div>Odds</div></div></div>'''


def test_every_raw_in_window_market_is_published_or_explicitly_excluded(tmp_path):
    scrape = ScrapeCoverage(tmp_path, "ncaaf")
    scrape.page(1, "https://example.test/dk", header())
    scrape.page(2, "https://example.test/dk?page=2", header("future", "9/20, 10:00PM"))
    rows = dk_rows()
    scrape.validation(rows)
    scrape.finish("COMPLETE")
    publication = PublicationCoverage(tmp_path, NOW)
    publication.reasons[("ncaaf", "123", "TOTAL")] = "INSUFFICIENT_HISTORY"
    board_path = tmp_path / "anomaly_board.csv"
    board_path.write_text("sport,game_id,market_display\n")
    summary = publication.publish(pd.DataFrame(), board_path, filter_publication_eligible_markets)
    assert summary["sports"]["ncaaf"]["eligible_in_window"] == 1
    assert summary["sports"]["ncaaf"]["excluded_by_reason"] == {"INSUFFICIENT_HISTORY": 1}
    assert summary["sports"]["ncaaf"]["outside_publication_window"] == 1
    # A later valid capture can publish the same discovered identity without
    # either weakening history or expanding the future-game horizon.
    later = PublicationCoverage(tmp_path, NOW)
    board = pd.DataFrame([dict(sport="ncaaf", game_id="123", market_display="TOTAL")])
    later.gate_ready = keys(board)
    board.to_csv(board_path, index=False)
    summary = later.publish(board, board_path, filter_publication_eligible_markets)
    assert summary["sports"]["ncaaf"]["published"] == 1
    assert summary["sports"]["ncaaf"]["excluded_by_reason"] == {}
    assert len(CoverageStore(tmp_path).inventory()) == 2


def test_stale_discovery_remains_durable_and_is_not_published(tmp_path):
    scrape = ScrapeCoverage(tmp_path, "ncaaf")
    scrape.page(1, "https://example.test/dk", header())
    scrape.validation(dk_rows()); scrape.finish("COMPLETE")
    publication = PublicationCoverage(tmp_path, NOW)
    key = ("ncaaf", "123", "TOTAL")
    publication.last_captures[key] = "2026-09-05T17:40:00Z"
    publication.reasons[key] = "STALE_CAPTURE"
    path = tmp_path / "anomaly_board.csv"; path.write_text("sport,game_id,market_display\n")
    summary = publication.publish(pd.DataFrame(), path, filter_publication_eligible_markets)
    assert summary["sports"]["ncaaf"]["excluded_by_reason"] == {"STALE_CAPTURE": 1}
    stored = CoverageStore(tmp_path).inventory()[0]
    assert stored["capture_age_minutes"] == 20
    assert stored["state"] == "STALE_CAPTURE"
    assert not stored["published"]


def test_unparsed_raw_headers_and_interrupted_scrapes_survive(tmp_path):
    scrape = ScrapeCoverage(tmp_path, "ncaaf")
    scrape.page(1, "https://example.test/dk", header())
    inventory = CoverageStore(tmp_path).inventory()
    assert inventory[0]["capture_exclusion_reason"] == "RAW_MARKET_PARSE_FAILED"
    with scrape.store.connect() as db:
        assert db.execute("SELECT state FROM runs").fetchone()[0] == "RUNNING"
        assert db.execute("SELECT count(*) FROM pages").fetchone()[0] == 1


def test_monitor_missing_evidence_is_not_success(tmp_path):
    result, healthy = check(tmp_path)
    assert not healthy
    assert result["issues"] == ["COVERAGE_EVIDENCE_UNAVAILABLE"]


def test_actual_publication_pipeline_accounts_for_all_in_window_markets(tmp_path, monkeypatch):
    import refresh_anomaly_board as refresh
    monkeypatch.setattr(refresh, "DATA", tmp_path)
    monkeypatch.setattr(refresh, "update_action_ledger", lambda *args: 0)
    monkeypatch.setattr(refresh, "apply_recorded_signals", lambda frame, *args: frame)
    monkeypatch.setattr(refresh, "rebuild_action_results", lambda *args: 0)
    records = []
    for gid, times, kick in [
        ("ready", ["17:55:00", "17:59:00"], KICK),
        ("stale", ["17:35:00", "17:40:00"], KICK),
        ("new", ["17:59:00"], KICK),
        ("future", ["17:55:00", "17:59:00"], "2026-09-20T02:00:00Z"),
    ]:
        for time in times:
            for row in dk_rows():
                records.append({**row, "game_id": gid, "timestamp": f"2026-09-05T{time}Z",
                                "current_line": row["current"], "open_line": row["current"], "dk_start_iso": kick})
    pd.DataFrame(records).to_csv(tmp_path / "snapshots.csv", index=False)
    coverage = PublicationCoverage(tmp_path, NOW)
    refresh._refresh(coverage)
    board = pd.read_csv(tmp_path / "anomaly_board.csv")
    assert set(board.game_id) == {"ready"}
    summary = json.loads((tmp_path / "publication_coverage.json").read_text())
    stats = summary["sports"]["ncaaf"]
    assert stats["eligible_in_window"] == 3
    assert stats["published"] == 1
    assert stats["excluded_by_reason"] == {"INSUFFICIENT_HISTORY": 1, "STALE_CAPTURE": 1}
    assert stats["outside_publication_window"] == 1
    assert summary["unexplained_gaps"] == 0
    # Version mismatch is detected independently of the source census warning.
    (tmp_path / "anomaly_board.csv").write_text("changed")
    result, healthy = check(tmp_path, now=NOW.to_pydatetime())
    assert not healthy and "COVERAGE_EXPORT_VERSION_MISMATCH" in result["issues"]


def test_empty_first_run_still_accounts_for_raw_discoveries(tmp_path, monkeypatch):
    import refresh_anomaly_board as refresh
    scrape = ScrapeCoverage(tmp_path, "ncaaf")
    scrape.page(1, "https://example.test/dk", header())
    monkeypatch.setattr(refresh, "DATA", tmp_path)
    monkeypatch.setattr(refresh, "update_action_ledger", lambda *args: 0)
    monkeypatch.setattr(refresh, "apply_recorded_signals", lambda frame, *args: frame)
    monkeypatch.setattr(refresh, "rebuild_action_results", lambda *args: 0)
    refresh._refresh(PublicationCoverage(tmp_path, NOW))
    summary = json.loads((tmp_path / "publication_coverage.json").read_text())
    assert summary["sports"]["ncaaf"]["excluded_by_reason"] == {"RAW_MARKET_PARSE_FAILED": 1}
    assert json.loads((tmp_path / "freshness.json").read_text())["board_market_count"] == 0


def test_scraper_preserves_discovery_when_split_parsing_produces_no_rows(tmp_path, monkeypatch):
    import dk_headless
    monkeypatch.setattr(dk_headless, "fetch_server_rendered_html", lambda *args: header())
    monkeypatch.setattr(dk_headless, "fetch_rendered_html", lambda *args, **kwargs: header())
    observer = ScrapeCoverage(tmp_path, "ncaaf")
    result = dk_headless.get_splits("https://example.test/dk", "ncaaf", coverage=observer)
    assert not result["records"]
    assert len(CoverageStore(tmp_path).inventory()) == 1
    with observer.store.connect() as db:
        assert db.execute("SELECT state FROM runs WHERE kind='SCRAPE'").fetchone()[0] == "RAW_MARKET_PARSE_FAILED"


@pytest.mark.parametrize("kickoff,inside", [("2026-11-09T04:59:59Z", True), ("2026-11-09T05:00:00Z", False)])
def test_dst_fix_does_not_publish_a_ninth_local_date(kickoff, inside):
    probe = pd.DataFrame([dict(sport="ncaaf", dk_start_iso=kickoff)])
    assert bool(len(filter_publication_eligible_markets(probe, now="2026-11-01T12:00:00-05:00"))) == inside


def test_wrong_side_identity_is_quarantined_without_deleting_valid_other_markets(monkeypatch):
    monkeypatch.setattr("main.get_espn_kickoff_map", lambda *args: {})
    valid = dk_rows()
    bad = dict(valid[0], side="Nevada", current="Nevada @ -150")
    accepted, _ = validate_snapshot_rows([*valid, bad], "ncaaf")
    assert accepted == valid
    assert bad["_capture_exclusion_reason"] == "SIDE_IDENTITY_MISMATCH"


def test_monitor_can_certify_a_fresh_complete_consistent_run(tmp_path):
    store = CoverageStore(tmp_path)
    for sport in ["mlb", "nfl", "ncaaf", "ufc"]:
        run_id = store.begin("SCRAPE", sport, now=NOW.isoformat())
        store.finish(run_id, "COMPLETE")
    store.update([dict(sport="ncaaf", game_id="123", market_display="TOTAL", game="UNLV @ Hawaii",
                       dk_start_iso=KICK, discovery_basis="RAW_DK_HEADER", league_identified=True,
                       capture_exclusion_reason="")], run_id, "CAPTURED")
    publication = PublicationCoverage(tmp_path, NOW)
    board = pd.DataFrame([dict(sport="ncaaf", game_id="123", market_display="TOTAL")])
    path = tmp_path / "anomaly_board.csv"; board.to_csv(path, index=False)
    publication.gate_ready = keys(board)
    publication.publish(board, path, filter_publication_eligible_markets)
    result, healthy = check(tmp_path, now=NOW.to_pydatetime())
    assert healthy, result["issues"]


def test_unchanged_publication_does_not_duplicate_historical_state_transitions(tmp_path):
    store = CoverageStore(tmp_path)
    row = dict(sport="ncaaf", game_id="123", market_display="TOTAL", state="STALE_CAPTURE",
               published=False, publication_eligible=False, in_window_scope=True)
    store.update([row], "first", "PUBLICATION_ACCOUNTED")
    store.update([{**row, "as_of": "later"}], "second", "PUBLICATION_ACCOUNTED")
    store.update([{**row, "state": "PUBLISHED", "published": True}], "third", "PUBLICATION_ACCOUNTED")
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM transitions").fetchone()[0] == 2
