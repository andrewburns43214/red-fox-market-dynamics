import json
from datetime import datetime, timedelta, timezone

import prop_projection_service as service
from prop_projection_prospective import build_candidate, freeze_candidate, frozen_candidates, grade_candidate, performance_summary


NOW = datetime(2026, 9, 21, 20, 0, tzinfo=timezone.utc)
START = NOW + timedelta(minutes=30)


def _outcome(player, name, price):
    return {"description": player, "name": name, "price": price, "last_seen_at": NOW.isoformat()}


def _fixture():
    event = {"id": "future", "away_team": "Away", "home_team": "Home", "commence_time": START.isoformat(),
             "bookmakers": [{"key": "pinnacle", "markets": [{"key": "player_anytime_td", "outcomes": [
                 _outcome("Away Back", "Over", -120), _outcome("Away Back", "Under", 100),
                 _outcome("Home Back", "Over", -110), _outcome("Home Back", "Under", -110)]}]},
                 {"key": "draftkings", "markets": [{"key": "player_2plus_td", "outcomes": [
                     _outcome("Away Back", "Yes", 450), _outcome("Home Back", "Yes", 400)]}]}]}
    roster = {"Away": ["Away Back"], "Home": ["Home Back"],
              "Away__positions": {"awayback": "RB"}, "Home__positions": {"homeback": "RB"}}
    lines = [{"team": team, "player_key": key, "market": "player_rush_yds", "mean": 80,
              "oldest_observed_at": NOW.isoformat()} for team, key in (("Away", "awayback"), ("Home", "homeback"))]
    production = {"status": "AVAILABLE", "event_id": "future", "model_version": "v1", "confidence": "MODERATE",
                  "away_mean": 21.25, "home_mean": 24.75, "audit_hash": "hash",
                  "coverage": {"teams": {team: {"scoring": {"rushing_td_source": "RUSH_YARDS_PROXY"}} for team in ("Away", "Home")}},
                  "components": {team: {"passing_td": 1.5, "rushing_td": 0.87, "kicking": 7.0} for team in ("Away", "Home")}}
    shadow = {"status": "SHADOW_AVAILABLE", "model_version": "v2", "confidence": "MODERATE",
              "baseline_variant": "consensus_candidate", "components": {},
              "variants": {"consensus_candidate": {"away_mean": 22.1, "home_mean": 25.2}}}
    return event, production, lines, shadow, roster


def test_four_methods_freeze_and_grade_without_postkickoff_mutation(tmp_path):
    args = _fixture()
    first = build_candidate(*args, NOW)
    assert set(first["methods"]) == set("ABCD")
    assert all(first["methods"][name]["status"] == "AVAILABLE" for name in "ABCD")
    assert first["methods"]["A"]["away_mean"] == 21.25
    assert first["methods"]["B"]["coverage"]["Away"]["production_rush_source"] == "RUSH_YARDS_PROXY"
    assert first["methods"]["B"]["components"]["Away"]["rushing_td"] > 0
    assert first["methods"]["C"]["components"]["Away"]["rushing_allocation"] == "UNRESOLVED"
    assert freeze_candidate(tmp_path, *args, NOW)
    assert not freeze_candidate(tmp_path, *args, NOW)
    path = tmp_path / "prospective_frozen" / "nfl_future.json"
    before = path.read_bytes()
    later = START + timedelta(minutes=1)
    assert not freeze_candidate(tmp_path, *args, later)
    assert path.read_bytes() == before
    candidates = frozen_candidates(tmp_path, later, {})
    assert list(candidates) == ["nfl:future"]
    graded = grade_candidate(candidates["nfl:future"], 17, 28, later)
    assert graded["methods"]["A"]["team_score_mae"] == 3.75
    assert graded["methods"]["A"]["game_total_error"] == 1.0
    assert graded["methods"]["A"]["scoring_margin_error"] == -7.5
    assert graded["methods"]["A"]["source_age_minutes_at_freeze"] == 0
    assert not frozen_candidates(tmp_path, later, {"nfl:future": graded})
    assert performance_summary({"games": {"nfl:future": graded}}, later)["metrics"]["B"]["games"] == 1


def test_stale_scorer_surface_marks_b_and_c_unavailable():
    event, production, lines, shadow, roster = _fixture()
    event["bookmakers"] = []
    candidate = build_candidate(event, production, lines, shadow, roster, NOW)
    assert candidate["methods"]["A"]["status"] == "AVAILABLE"
    assert candidate["methods"]["B"]["status"] == "UNAVAILABLE"
    assert candidate["methods"]["C"]["status"] == "UNAVAILABLE"
    assert candidate["methods"]["D"]["status"] == "AVAILABLE"


def test_last_available_method_survives_later_missing_quotes(tmp_path):
    event, production, lines, shadow, roster = _fixture()
    assert freeze_candidate(tmp_path, event, production, lines, shadow, roster, NOW)
    event["bookmakers"] = []
    assert freeze_candidate(tmp_path, event, production, lines, shadow, roster, NOW + timedelta(minutes=5))
    saved = json.loads((tmp_path / "prospective_frozen" / "nfl_future.json").read_text())
    assert saved["methods"]["B"]["status"] == "AVAILABLE"
    assert saved["methods"]["B"]["frozen_at"] == NOW.isoformat()
    assert saved["methods"]["A"]["frozen_at"] == (NOW + timedelta(minutes=5)).isoformat()


def test_final_score_resolver_grades_frozen_file_once(monkeypatch, tmp_path):
    args = _fixture()
    freeze_candidate(tmp_path, *args, NOW)
    monkeypatch.setattr(service, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(service, "SPORTS", {"nfl": {"enabled": True, "provider_key": "football_nfl"}})
    class Scores:
        calls = 0
        def scores(self, sport):
            self.calls += 1
            return [{"id": "future", "status": "final", "away_score": 17, "home_score": 28}]
    client = Scores()
    service.run_resolution(client=client, now=START + timedelta(hours=1), force=True)
    grades = json.loads((tmp_path / "prospective_grades.json").read_text())["games"]
    assert list(grades) == ["nfl:future"]
    assert all(grades["nfl:future"]["methods"][method]["status"] == "GRADED" for method in "ABCD")
    first = (tmp_path / "prospective_grades.json").read_bytes()
    service.run_resolution(client=client, now=START + timedelta(hours=2), force=True)
    assert (tmp_path / "prospective_grades.json").read_bytes() == first
    assert client.calls == 1
