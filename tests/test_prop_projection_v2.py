import inspect
import math
from datetime import datetime, timezone

import pytest

import prop_projection_service as service
import prop_projection_v2
from prop_projection_config import SPORTS
from prop_projection_service import _shadow_performance
from prop_projection_v2 import convert_line, poisson_mean_from_over, poisson_survival, project_event_v2
from prop_projection_v2_config import BINARY_EVENT, COUNT, DISTRIBUTION_KIND, NFL_HISTORICAL_BASELINE


NOW = datetime(2026, 9, 20, 16, 0, tzinfo=timezone.utc)


def line(team, player, market, point, probability=0.31, books=3):
    return {
        "team": team,
        "player": player,
        "player_key": player.lower().replace(" ", ""),
        "player_id": f"id:{player}",
        "market": market,
        "family": "test",
        "classification": "CORE",
        "line": point,
        "mean": point,
        "fair_over": probability,
        "books": [f"book-{index}" for index in range(books)],
        "book_count": books,
        "observed_at": NOW.isoformat(),
        "selection_audit": {},
    }


def v1(sport="nfl", confidence="HIGH"):
    return {
        "sport": sport,
        "event_id": "event-1",
        "away_team": "Away",
        "home_team": "Home",
        "commence_time": "2026-09-21T00:00:00+00:00",
        "generated_at": NOW.isoformat(),
        "model_version": f"prop_projection_{sport}_v1",
        "audit_hash": "same-lines",
        "status": "AVAILABLE",
        "confidence": confidence,
        "away_score": 20,
        "home_score": 21,
        "away_mean": 20.1,
        "home_mean": 20.9,
    }


def test_every_configured_market_has_an_explicit_distribution_class():
    configured = {market for config in SPORTS.values() for market in config["markets"]}
    assert configured <= set(DISTRIBUTION_KIND)


def test_poisson_tail_inversion_reproduces_price_and_obeys_mean_bound():
    probability = 0.31
    mean = poisson_mean_from_over(0.5, probability)
    assert poisson_survival(mean, 1) == pytest.approx(probability, abs=1e-8)
    assert mean >= probability
    converted = convert_line(line("Away", "Batter", "batter_rbis", 0.5, probability))
    assert converted["v2_kind"] == COUNT
    assert converted["v2_mean"] >= probability


def test_binary_event_uses_devigged_probability_not_continuous_line():
    converted = convert_line(line("Away", "Slugger", "batter_home_runs", 0.5, 0.22))
    assert converted["v2_kind"] == BINARY_EVENT
    assert converted["v2_distribution"] == "bernoulli_event"
    assert converted["v2_mean"] == pytest.approx(0.22)


def test_nfl_fg_only_one_book_is_not_high_and_uses_historical_kicking():
    lines = []
    for team in ("Away", "Home"):
        lines.extend([
            line(team, f"{team} QB", "player_pass_tds", 1.5, 0.5, 4),
            line(team, f"{team} RB", "player_rush_yds", 65.5, 0.5, 3),
            line(team, f"{team} K", "player_field_goals_made", 1.5, 0.5, 1),
        ])
    result = project_event_v2("nfl", v1(), lines, now=NOW)
    assert result["status"] == "SHADOW_AVAILABLE"
    assert result["confidence"] == "MODERATE"
    assert result["components"]["Away"]["selected_kicking_method"] == "league_kicking_baseline"
    assert result["components"]["Away"]["selected_kicking"] == NFL_HISTORICAL_BASELINE["kicking_points_per_team_game"]
    assert "rushing_yards_per_92_unvalidated" in result["variants"]
    assert "league_rushing_td_rate" in result["variants"]
    assert result["components"]["Away"]["residual_scoring"] > 0


def test_nfl_scorer_td_surface_is_an_alternative_not_double_counted():
    lines = []
    for team in ("Away", "Home"):
        lines.extend([
            line(team, f"{team} QB", "player_pass_tds", 1.5, 0.5, 4),
            line(team, f"{team} RB", "player_rush_tds", 0.5, 0.35, 3),
            line(team, f"{team} WR", "player_reception_tds", 0.5, 0.40, 3),
            line(team, f"{team} K", "player_kicking_points", 6.5, 0.5, 3),
        ])
    result = project_event_v2("nfl", v1(), lines, now=NOW)
    assert "scorer_td_surface" in result["variants"]
    components = result["components"]["Away"]
    scorer_mean = components["offensive_td_candidates"]["scorer_td_surface"]
    expected = 6 * scorer_mean + components["selected_kicking"] + components["residual_scoring"]
    assert result["variants"]["scorer_td_surface"]["away_mean"] == pytest.approx(expected, abs=1e-4)


def test_mlb_preserves_independent_estimators_and_multiple_combinations():
    lines = []
    for team in ("Away", "Home"):
        for index in range(9):
            player = f"{team} Batter {index}"
            lines.extend([
                line(team, player, "batter_runs", 0.5, 0.31, 2),
                line(team, player, "batter_rbis", 0.5, 0.28, 2),
                line(team, player, "batter_hits", 0.5, 0.57, 3),
                line(team, player, "batter_total_bases", 1.5, 0.45, 3),
                line(team, player, "batter_home_runs", 0.5, 0.12, 2),
                line(team, player, "batter_walks", 0.5, 0.27, 2),
            ])
        lines.extend([
            line(team, f"{team} Pitcher", "pitcher_earned_runs", 2.5, 0.45, 4),
            line(team, f"{team} Pitcher", "pitcher_outs", 17.5, 0.5, 5),
        ])
    context = {"away_probable_pitcher": "Away Pitcher", "home_probable_pitcher": "Home Pitcher"}
    result = project_event_v2("mlb", v1("mlb"), lines, context=context, now=NOW)
    assert result["status"] == "SHADOW_AVAILABLE"
    assert {item["name"] for item in result["components"]["Away"]} == {
        "batter_expected_runs", "batter_expected_rbi", "offensive_linear_weights", "starter_er_plus_bullpen",
    }
    assert {"simple_median", "quality_weighted_median", "robust_quality_weighted_mean", "consensus_candidate"} <= set(result["variants"])


def test_shadow_performance_compares_every_variant_to_actual():
    rows = [{
        "sport": "nfl", "actual_away": 20, "actual_home": 24,
        "v1_benchmark": {"away_score": 18, "home_score": 21},
        "variants": {
            "a": {"away_score": 21, "home_score": 23},
            "b": {"away_score": 17, "home_score": 24},
        },
    }]
    metrics = _shadow_performance(rows)["nfl"]
    assert metrics["a"]["team_score_mae"] == 1.0
    assert metrics["a"]["total_bias"] == 0.0
    assert metrics["b"]["decisive_winner_accuracy"] == 1.0
    assert metrics["v1_customer_model"]["games"] == 1


def test_shadow_backfill_uses_private_v1_ledger_without_public_write(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "DATA_ROOT", tmp_path)
    source = {
        "projection": v1(),
        "canonical_lines": [
            line("Away", "Away QB", "player_pass_tds", 1.5, 0.5, 3),
            line("Away", "Away RB", "player_rush_yds", 60.5, 0.5, 3),
            line("Away", "Away K", "player_kicking_points", 6.5, 0.5, 3),
            line("Home", "Home QB", "player_pass_tds", 1.5, 0.5, 3),
            line("Home", "Home RB", "player_rush_yds", 60.5, 0.5, 3),
            line("Home", "Home K", "player_kicking_points", 6.5, 0.5, 3),
        ],
    }
    (tmp_path / "projection_ledger.jsonl").write_text(__import__("json").dumps(source) + "\n")
    (tmp_path / "state.json").write_text("{}")
    result = service.run_shadow_backfill()
    assert result == {"appended": 1, "skipped": 0, "failures": 0}
    assert (tmp_path / "projection_v2_shadow_ledger.jsonl").exists()
    assert not (tmp_path / "prop_projections.json").exists()
    assert service.run_shadow_backfill()["skipped"] == 1


def test_v2_is_shadow_only_and_has_no_game_market_input():
    source = inspect.getsource(prop_projection_v2).lower()
    assert '"customer_facing": false' in source
    for forbidden in ("moneyline", "game_total", "market_read", "red_fox_favorite", "anomaly_board"):
        assert forbidden not in source
