from datetime import datetime, timezone
import json
import os

import pytest

from history_evidence import Ledger
from history_scenarios import FOOTBALL_KEYS, key_path, partitions, reliable_splits, role
from history_team_evidence import TeamEvidence


def snapshot(opening, observed, tickets=71, money=89, *, price=-110, peer=-.04, total=-2):
    return dict(open_line=opening, line=observed, line_toward=opening-observed, open_price=-110, price=price,
                tickets=tickets, money=money, side="selected", price_toward_implied=0.0,
                game_context={"MONEYLINE": {"price_toward_implied": peer},
                              "TOTAL": {"open_line": 48.5, "line": 48.5+total}})


def opposing(quote):
    return dict(open_line=-quote["open_line"], line=-quote["line"],
                tickets=100-quote["tickets"], money=100-quote["money"], price=-110, side="opponent")


def populate(ledger, index, quote, grade):
    row = dict(sport="ncaaf", event_id=str(index), matchup="Away @ Home",
               scheduled_at="2026-10-02T23:00:00Z", entry_at="2026-10-02T22:00:00Z",
               sides=(quote, opposing(quote)))
    for group in ("closest", "broader"):
        ledger.add(group, row, quote, "Away", grade, "Away 23 · Home 16", [], .30)


def test_same_role_closer_openers_and_key_path_keep_weak_examples_related():
    current = snapshot(-6, -4.5)
    ledger = Ledger()
    populate(ledger, 1, snapshot(-6, -4.5, price=-120), "win")
    populate(ledger, 2, snapshot(-3, -1.5, 65, 84), "loss")
    populate(ledger, 3, snapshot(7.5, 9.5, 71, 77), "push")
    populate(ledger, 4, snapshot(-6, -4.5, 0, 0), "win")
    groups = {}
    partitions(ledger, groups, "ncaaf", "SPREAD", current, opposing(current))
    assert ledger.groups["best"] == [0]
    assert ledger.groups["related"] == [1, 2]
    assert ledger.groups["scenario_confirmation"] == [0, 1]
    assert groups["best"]["win"] == 1
    assert groups["best"]["odds"] == -120
    assert groups["best"]["roi"] == pytest.approx(83.33)
    assert groups["related"]["graded"] == 2
    assert groups["related"]["roi"] == -50
    assert groups["related"]["loss"] == 1 and groups["related"]["push"] == 1


def test_scenarios_do_not_use_wins_or_choose_an_outcome_when_evidence_missing():
    current = snapshot(-6, -4.5, peer=None)
    ledger = Ledger()
    populate(ledger, 1, snapshot(-6, -4.5), "loss")
    first = {}
    partitions(ledger, first, "ncaaf", "SPREAD", current, opposing(current))
    ledger.rows[0]["grade"] = "win"
    second = {}
    partitions(ledger, second, "ncaaf", "SPREAD", current, opposing(current))
    assert first["scenarios"] == second["scenarios"]
    assert not any(item["key"] == "scenario_confirmation" for item in second["scenarios"])
    assert first["best"]["roi"] != second["best"]["roi"]


def test_football_endpoint_classification_and_moneyline_role_are_not_price_sign_shortcuts():
    assert key_path(snapshot(-6, -4.5), FOOTBALL_KEYS) == ((-6, "off"),)
    assert key_path(snapshot(3.5, 3), FOOTBALL_KEYS) == ((3, "onto"),)
    assert key_path(snapshot(3.5, 2.5), FOOTBALL_KEYS) == ((3, "through"),)
    assert role("MONEYLINE", {"open_price": -105, "price": -120}, {"open_price": -120, "price": +100}) == (-1, 1)
    assert reliable_splits({"tickets": 100, "money": 78}, {"tickets": 0, "money": 22})
    assert not reliable_splits({"tickets": 100, "money": 100}, {"tickets": 0, "money": 0})
    assert not reliable_splits({"tickets": 71, "money": 89}, {"tickets": 33, "money": 11})


def test_private_team_context_requires_the_same_fixture_clock_and_prior_verified_results(tmp_path):
    def stamp(day):
        return datetime(2026, 11, day, tzinfo=timezone.utc).isoformat(timespec="microseconds")

    path = tmp_path / "nfl.json"
    path.write_text(json.dumps({"generated": stamp(24), "fixtures": {
        "event": {"start": stamp(27), "keys": ["away"], "sporting": {"independent": {"record": "8–2"}}}},
        "teams": {"away": ["early", "later"]}, "games": {
        "early": {"start": stamp(22), "source_at": stamp(23), "scores": [14, 7]},
        "later": {"start": stamp(22), "source_at": stamp(26), "scores": [7, 14]}}}), encoding="utf-8")
    os.utime(path, (datetime.fromisoformat(stamp(24)).timestamp(),) * 2)
    reader = TeamEvidence(tmp_path)
    receipt = reader.capture("nfl", "event", stamp(25), stamp(27))
    assert receipt["verified_recent"]["away"][0]["scores"] == [14, 7]
    assert len(receipt["verified_recent"]["away"]) == 1
    assert reader.capture("nfl", "event", stamp(23), stamp(27)) is None
    assert reader.capture("nfl", "event", stamp(25), stamp(28)) is None
