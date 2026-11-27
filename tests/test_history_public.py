import csv
from datetime import datetime, timedelta, timezone
import json

from history_blocks import merge
from history_comparisons import cross_evidence, recorded_split_context
from history_monitor import GIB, forecast
from history_public import History, build, decision_window, distance, line_price, query_features
from history_store import connect, store_outcome, utc


def side(name, price=-110, line=3):
    return {"side": name, "line": line, "price": price, "open_line": 4,
            "open_price": -110, "tickets": 25, "money": 20, "line_toward": 1,
            "price_toward_implied": 0}


def test_nearest_scores_market_evidence_not_results_and_limits_market_types():
    query = side("home")
    closer = side("away") | {"result": "loss"}
    wider = side("away") | {"tickets": 41, "money": 35, "result": "win"}
    assert distance(query, closer, "nfl", "SPREAD")[0] < distance(query, wider, "nfl", "SPREAD")[0]
    assert distance(query, closer | {"result": "win"}, "nfl", "SPREAD") == distance(query, closer, "nfl", "SPREAD")
    assert distance(side("over"), side("under"), "mlb", "TOTAL") is None
    assert line_price("+6 (-110)", "SPREAD") == (6, -110)
    assert line_price("O 8.5 (+111)", "TOTAL") == (8.5, 111)
    assert line_price("-155", "MONEYLINE") == (None, -155)
    assert decision_window(32) == 30 and decision_window(7000) is None


def test_report_uses_individual_prices_asof_results_and_unique_fixtures(tmp_path):
    with connect(tmp_path) as db:
        ranked = []
        for event, scores, price, provider, observed in (
            ("one", (14, 10), 150, "1", "2026-10-20T00:00:00Z"),
            ("duplicate", (14, 10), -110, "1", "2026-10-20T00:00:00Z"),
            ("two", (20, 7), -150, "2", "2026-10-20T00:00:00Z"),
            ("three", (17, 20), -100, "3", "2026-10-20T00:00:00Z"),
            ("future", (14, 10), -110, "4", "2026-12-30T00:00:00Z"),
        ):
            game = f"Away {event} @ Home {event}"
            store_outcome(db, "nfl", event, "verified", "verified",
                          {"provider_event_id": provider, "observed_at": observed,
                           "matchup": game, "score_away": scores[0], "score_home": scores[1]})
            ranked.append((0, event, {"sport": "nfl", "matchup": game, "scheduled_at": "2026-10-19T00:00:00Z",
                                    "entry_at": "2026-10-18T23:30:00Z"}, side(f"away {event}", price, -3),
                           True, (("Ticket share", 0, 1),)))
        # Give the first side a win, the second a loss, and the third a push at its recorded line.
        ranked[2][3]["line"] = -15
        ranked[3][3]["line"] = 3
        result = History(db).report(ranked, "SPREAD", utc("2026-11-27T00:00:00Z"))
        assert (result["matched"], result["graded"], result["win"], result["loss"], result["push"]) == (4, 3, 1, 1, 1)
        assert result["unresolved"] == 1 and result["units"] == .5 and result["roi"] == 16.67
        assert result["odds"] == 106
        assert [example["grade"] for example in result["examples"]] == ["win", "loss", "push"]


def board_row(now):
    observed = utc(now - timedelta(minutes=4))
    sides = [{"flagged_side": name, "source_latest_at": observed, "open_line": opening,
              "current_line": current, "bets_pct": pct, "money_pct": pct, "data_badge": "Clean"}
             for name, opening, current, pct in (
                 ("NY Giants -6", "-6.5 (-110)", "-6 (-115)", 80),
                 ("IND Colts +6", "+6.5 (-110)", "+6 (-105)", 20),
             )]
    return {"sport": "nfl", "game_id": "test", "game": "NY Giants @ IND Colts", "market_display": "SPREAD",
            "source_latest_at": observed, "kickoff_iso": utc(now + timedelta(minutes=26)),
            "market_sides": json.dumps(sides), "red_fox_favorite": "true", "favorite_side": "IND Colts +6"}


def write_csv(path, row):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)


def test_board_side_identity_and_frozen_context_does_not_recompute(tmp_path):
    now = datetime(2026, 11, 27, 20, tzinfo=timezone.utc)
    data = tmp_path / "data"
    data.mkdir()
    output = tmp_path / "public"
    row = board_row(now)
    with connect(tmp_path / "db") as db:
        setup = query_features(db, {"SPREAD": row}, row, False, now)
        assert [value["name"] for value in setup["sides"]] == ["NY Giants", "IND Colts"]
        assert [value["favorite"] for value in setup["sides"]] == [False, True]
        assert setup["sides"][1]["opening_money"] is None
        write_csv(data / "anomaly_board.csv", row)
        assert build(db, tmp_path, output, tmp_path / "disabled", now)["games"] == 1
        game = output / "nfl--test.json"
        write_csv(data / "live_recent.csv", row | {"frozen_at_utc": row["kickoff_iso"]})
        first = now + timedelta(minutes=45)
        assert build(db, tmp_path, output, tmp_path / "disabled", first)["frozen"] == 1
        frozen = game.read_bytes()
        new = row | {"favorite_side": "", "frozen_at_utc": row["kickoff_iso"]}
        write_csv(data / "live_recent.csv", new)
        assert build(db, tmp_path, output, tmp_path / "disabled", first + timedelta(hours=2))["frozen"] == 0
        assert game.read_bytes() == frozen


def test_frozen_one_side_requires_the_exact_paired_archived_quote(tmp_path):
    now = datetime(2026, 11, 27, 20, tzinfo=timezone.utc)
    row = board_row(now)
    shown = json.loads(row["market_sides"])[1:]
    row["market_sides"] = json.dumps(shown)
    capture, schedule = row["source_latest_at"], row["kickoff_iso"]
    with connect(tmp_path) as db:
        assert query_features(db, {}, row, True, now) is None
        merge(db, [("nfl", "test", "SPREAD", capture, side, schedule, line, price, opener, -110, split, split, "{}")
                   for side, line, price, opener, split in (
                           ("ny giants", -6, -115, -6.5, 80), ("indianapolis colts", 6, -105, 6.5, 20))])
        setup = query_features(db, {}, row, True, now)
        assert [(item["name"], item["price"]) for item in setup["sides"]] == [
            ("NY Giants", -115), ("IND Colts", -105)]
        legacy = board_row(now)
        old_sides = [dict(item, source_latest_at=None) for item in json.loads(legacy["market_sides"])]
        old_sides[0]["current_line"], old_sides[1]["current_line"] = "NY Giants -6 @ -115", "IND Colts +6 @ -105"
        old_sides[0]["open_line"], old_sides[1]["open_line"] = "NY Giants -6.5 @ -110", "IND Colts +6.5 @ -110"
        legacy.update(source_latest_at="", market_sides=json.dumps(old_sides), frozen_at_utc=schedule)
        assert query_features(db, {}, legacy, True, now)["observed"] == capture
        old_sides[1]["money_pct"] = 21
        assert query_features(db, {}, legacy | {"market_sides": json.dumps(old_sides)}, True, now) is None
        shown[0]["current_line"] = "+6 (-107)"
        assert query_features(db, {}, row | {"market_sides": json.dumps(shown)}, True, now) is None


def test_split_audit_context_is_exact_side_and_point_in_time(tmp_path):
    with connect(tmp_path) as db:
        before = utc("2026-11-27T19:30:00Z")
        for known, category in (("19:25", "ticket_only"), ("19:35", "both")):
            stamp = utc("2026-11-27T" + known + ":00Z")
            document = {"review_state": "cleared", "decision_evidence": json.dumps({
                "split_breach": category, "phase": "reinstatement", "selected_side": "IND Colts +6",
                "source_observed_at": stamp,
            })}
            db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?)",
                       ("nfl", "test", "SPREAD", stamp, "red_fox_favorite_review", stamp, json.dumps(document)))
        assert recorded_split_context(db, "nfl", "test", "SPREAD", before) == {
            "indianapolis colts": {"split_breach": "ticket_only", "phase": "reinstatement"}}
        quote = {"open_line": None, "line": None, "open_price": 150, "price": 120}
        assert cross_evidence("MONEYLINE", quote, [quote | {"price": 130}])["sustained_toward"] is True
        assert cross_evidence("MONEYLINE", quote, [quote | {"price": 150}])["sustained_toward"] is False
        assert cross_evidence("MONEYLINE", quote)["sustained_toward"] is None


def test_storage_forecast_requires_sustained_daily_trend():
    points = {str(day): (30 - day / 50) * GIB for day in range(33)}
    assert forecast({"10": 21 * GIB}, 21 * GIB) is None
    assert abs(forecast(points, 20 * GIB) - 250) < .01


def test_replaced_ufc_opponent_cannot_grade_an_older_quote(tmp_path):
    with connect(tmp_path) as db:
        before = utc("2026-11-27T00:00:00Z")
        old_side = side("First Fighter", -110)
        row = {"sport": "ufc", "matchup": "First Fighter vs Old Opponent",
               "scheduled_at": utc("2026-11-20T20:00:00Z"), "entry_at": utc("2026-11-20T19:30:00Z"),
               "sides": [old_side, side("Old Opponent", -110)]}
        store_outcome(db, "ufc", "bout", "verified", "verified", {
            "provider_event_id": "replacement", "observed_at": "2026-11-20T23:00:00Z",
            "matchup": "First Fighter vs New Opponent", "winner": "First Fighter",
        })
        report = History(db).report([(0, "bout", row, old_side, True, ())], "MONEYLINE", before)
        assert (report["matched"], report["graded"], report["unresolved"]) == (1, 0, 1)
