"""Private v2 shadow projection built from v1's exact canonical prop surface.

Nothing in this module is published to the customer board. It exists to compare
mathematically valid distribution conversions and competing score estimators
against v1 and final results before any production-model promotion decision.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from statistics import NormalDist

from prop_projection_v2_config import (
    BINARY_EVENT,
    CONTINUOUS,
    CONTINUOUS_SIGMA,
    COUNT,
    DISTRIBUTION_KIND,
    MLB_BULLPEN_RUNS_PER_INNING,
    MLB_LINEAR_WEIGHTS,
    MLB_RBI_TO_RUNS,
    NFL_HISTORICAL_BASELINE,
    V2_MODEL_VERSION,
)


def poisson_survival(mean: float, minimum_count: int) -> float:
    """P[X >= minimum_count] for a Poisson random variable."""
    if minimum_count <= 0:
        return 1.0
    if mean <= 0:
        return 0.0
    term = math.exp(-mean)
    cumulative = term
    for value in range(1, minimum_count):
        term *= mean / value
        cumulative += term
    return min(1.0, max(0.0, 1.0 - cumulative))


def poisson_mean_from_over(line: float, fair_over: float) -> float:
    """Invert an Over price at an integer-count threshold into a Poisson mean."""
    minimum_count = math.floor(float(line)) + 1
    probability = min(1.0 - 1e-9, max(1e-9, float(fair_over)))
    low, high = 0.0, max(1.0, minimum_count * 2.0)
    while poisson_survival(high, minimum_count) < probability and high < 4096:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if poisson_survival(middle, minimum_count) < probability:
            low = middle
        else:
            high = middle
    mean = (low + high) / 2.0
    # For any nonnegative integer count, E[X] >= k * P(X >= k).
    return max(mean, minimum_count * probability)


def convert_line(line: dict) -> dict:
    """Attach a v2 mean without mutating the v1 canonical line."""
    market = line["market"]
    kind = DISTRIBUTION_KIND[market]
    threshold = float(line["line"])
    probability = min(0.999999, max(0.000001, float(line["fair_over"])))
    if kind == CONTINUOUS:
        sigma = CONTINUOUS_SIGMA[market]
        mean = max(0.0, threshold + sigma * NormalDist().inv_cdf(probability))
        distribution = "normal_quantile"
    elif kind == BINARY_EVENT:
        mean = probability
        distribution = "bernoulli_event"
    elif kind == COUNT:
        mean = poisson_mean_from_over(threshold, probability)
        distribution = "poisson_tail_inversion"
    else:  # pragma: no cover - guarded by the explicit configuration contract.
        raise ValueError(f"unknown_distribution_kind:{kind}")
    return {
        **line,
        "v2_kind": kind,
        "v2_distribution": distribution,
        "v2_mean": round(mean, 6),
        "v2_minimum_mean_bound": round((math.floor(threshold) + 1) * probability, 6) if kind == COUNT else None,
    }


def convert_lines(lines: list[dict]) -> list[dict]:
    return [convert_line(line) for line in lines if line.get("market") in DISTRIBUTION_KIND]


def _team_lines(lines, team, market):
    return [line for line in lines if line["team"] == team and line["market"] == market]


def _best(lines):
    return sorted(lines, key=lambda line: (-line["book_count"], line["player_key"]))[0] if lines else None


def _display_score(value):
    return int(round(min(60.0, max(0.0, value))))


def _nfl_team_components(lines, team):
    passing = _best(_team_lines(lines, team, "player_pass_tds"))
    pass_td = passing["v2_mean"] if passing else None

    direct_rush = _team_lines(lines, team, "player_rush_tds")
    receiving_tds = _team_lines(lines, team, "player_reception_tds")
    rush_yards = _team_lines(lines, team, "player_rush_yds")
    rushing_candidates = {
        "league_rushing_td_rate": NFL_HISTORICAL_BASELINE["rushing_td_per_team_game"],
    }
    if direct_rush:
        rushing_candidates["direct_rushing_td_props"] = sum(line["v2_mean"] for line in direct_rush)
    if rush_yards:
        rushing_candidates["rushing_yards_per_92_unvalidated"] = sum(line["v2_mean"] for line in rush_yards) / 92.0
    offensive_td_candidates = {}
    if direct_rush and receiving_tds:
        # This is an alternative to pass-TD + rush-TD, never an addition to it.
        # It can be evaluated when the scorer surface is sufficiently complete.
        offensive_td_candidates["scorer_td_surface"] = (
            sum(line["v2_mean"] for line in direct_rush)
            + sum(line["v2_mean"] for line in receiving_tds)
        )

    direct_kick = _best(_team_lines(lines, team, "player_kicking_points"))
    field_goals = _best(_team_lines(lines, team, "player_field_goals_made"))
    extra_points = _best(_team_lines(lines, team, "player_extra_points_made"))
    kicking_candidates = {
        "league_kicking_baseline": {
            "value": NFL_HISTORICAL_BASELINE["kicking_points_per_team_game"],
            "reliable": False,
        }
    }
    if direct_kick:
        kicking_candidates["direct_kicker_points"] = {
            "value": direct_kick["v2_mean"],
            "reliable": direct_kick["book_count"] >= 2,
            "book_count": direct_kick["book_count"],
        }
    if field_goals and extra_points:
        kicking_candidates["field_goals_plus_extra_points"] = {
            "value": 3.0 * field_goals["v2_mean"] + extra_points["v2_mean"],
            "reliable": field_goals["book_count"] >= 2 and extra_points["book_count"] >= 2,
            "book_count": min(field_goals["book_count"], extra_points["book_count"]),
        }

    if kicking_candidates.get("direct_kicker_points", {}).get("reliable"):
        kicking_method = "direct_kicker_points"
    elif kicking_candidates.get("field_goals_plus_extra_points", {}).get("reliable"):
        kicking_method = "field_goals_plus_extra_points"
    else:
        kicking_method = "league_kicking_baseline"

    return {
        "passing_td": pass_td,
        "passing_source": passing,
        "rushing_candidates": rushing_candidates,
        "offensive_td_candidates": offensive_td_candidates,
        "receiving_td_validation": sum(line["v2_mean"] for line in receiving_tds) if receiving_tds else None,
        "kicking_candidates": kicking_candidates,
        "selected_kicking_method": kicking_method,
        "selected_kicking": kicking_candidates[kicking_method]["value"],
        "kicking_reliable": kicking_method != "league_kicking_baseline",
        "residual_scoring": NFL_HISTORICAL_BASELINE["residual_points_per_team_game"],
    }


def _nfl_projection(sport, v1, lines):
    away, home = v1["away_team"], v1["home_team"]
    components = {team: _nfl_team_components(lines, team) for team in (away, home)}
    if any(components[team]["passing_td"] is None for team in (away, home)):
        return {"status": "SHADOW_UNAVAILABLE", "reason": "missing_passing_td_component"}

    common_methods = set(components[away]["rushing_candidates"]) & set(components[home]["rushing_candidates"])
    variants = {}
    for method in sorted(common_methods):
        means = {}
        for team in (away, home):
            item = components[team]
            means[team] = (
                6.0 * (item["passing_td"] + item["rushing_candidates"][method])
                + item["selected_kicking"]
                + item["residual_scoring"]
            )
        variants[method] = {
            "away_mean": round(means[away], 4),
            "home_mean": round(means[home], 4),
            "away_score": _display_score(means[away]),
            "home_score": _display_score(means[home]),
        }

    scorer_available = all("scorer_td_surface" in components[team]["offensive_td_candidates"] for team in (away, home))
    if scorer_available:
        means = {}
        for team in (away, home):
            item = components[team]
            means[team] = (
                6.0 * item["offensive_td_candidates"]["scorer_td_surface"]
                + item["selected_kicking"]
                + item["residual_scoring"]
            )
        variants["scorer_td_surface"] = {
            "away_mean": round(means[away], 4), "home_mean": round(means[home], 4),
            "away_score": _display_score(means[away]), "home_score": _display_score(means[home]),
        }

    consensus_means = {}
    for team in (away, home):
        item = components[team]
        rush = statistics.median(item["rushing_candidates"].values())
        consensus_means[team] = 6.0 * (item["passing_td"] + rush) + item["selected_kicking"] + item["residual_scoring"]
    variants["consensus_candidate"] = {
        "away_mean": round(consensus_means[away], 4),
        "home_mean": round(consensus_means[home], 4),
        "away_score": _display_score(consensus_means[away]),
        "home_score": _display_score(consensus_means[home]),
    }
    kicking_complete = all(components[team]["kicking_reliable"] for team in (away, home))
    confidence = "HIGH" if v1.get("confidence") == "HIGH" and kicking_complete else "MODERATE"
    return {
        "status": "SHADOW_AVAILABLE",
        "confidence": confidence,
        "baseline_variant": "consensus_candidate",
        "variants": variants,
        "components": components,
        "assumptions": {"nfl_historical_baseline": NFL_HISTORICAL_BASELINE},
    }


def _quality(lines, expected_players):
    if not lines:
        return 0.0
    players = len({line["player_key"] for line in lines})
    coverage = min(1.0, players / expected_players)
    depth = statistics.median(min(1.0, line["book_count"] / 3.0) for line in lines)
    return max(0.05, coverage * depth)


def _weighted_median(values):
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    cursor = 0.0
    for value, weight in ordered:
        cursor += weight
        if cursor >= total / 2.0:
            return value
    return ordered[-1][0]


def _robust_weighted_mean(values):
    raw = [value for value, _ in values]
    center = statistics.median(raw)
    deviations = [abs(value - center) for value in raw]
    mad = statistics.median(deviations) or 0.5
    radius = max(1.0, 2.5 * mad)
    clipped = [(min(center + radius, max(center - radius, value)), weight) for value, weight in values]
    return sum(value * weight for value, weight in clipped) / sum(weight for _, weight in clipped)


def _probable_pitcher_key(context, defense, away, home):
    field = "away_probable_pitcher" if defense == away else "home_probable_pitcher"
    raw = str(context.get(field) or "")
    return "".join(character for character in raw.lower() if character.isalnum())


def _mlb_team_estimators(lines, offense, defense, away, home, context):
    def market(key, team=offense):
        return _team_lines(lines, team, key)

    estimators = []
    batter_runs = market("batter_runs")
    batter_rbis = market("batter_rbis")
    hits, total_bases = market("batter_hits"), market("batter_total_bases")
    home_runs, walks = market("batter_home_runs"), market("batter_walks")

    if batter_runs:
        estimators.append({
            "name": "batter_expected_runs", "value": sum(line["v2_mean"] for line in batter_runs),
            "weight": _quality(batter_runs, 9), "inputs": len(batter_runs),
        })
    if batter_rbis:
        estimators.append({
            "name": "batter_expected_rbi", "value": MLB_RBI_TO_RUNS * sum(line["v2_mean"] for line in batter_rbis),
            "weight": _quality(batter_rbis, 9), "inputs": len(batter_rbis),
        })
    if hits and total_bases:
        value = (
            MLB_LINEAR_WEIGHTS["hits"] * sum(line["v2_mean"] for line in hits)
            + MLB_LINEAR_WEIGHTS["total_bases"] * sum(line["v2_mean"] for line in total_bases)
            + MLB_LINEAR_WEIGHTS["home_runs"] * sum(line["v2_mean"] for line in home_runs)
            + MLB_LINEAR_WEIGHTS["walks"] * sum(line["v2_mean"] for line in walks)
        )
        relevant = hits + total_bases + home_runs + walks
        estimators.append({
            "name": "offensive_linear_weights", "value": value,
            "weight": _quality(relevant, 27), "inputs": len(relevant),
        })

    pitcher_key = _probable_pitcher_key(context, defense, away, home)
    earned_candidates = market("pitcher_earned_runs", defense)
    out_candidates = market("pitcher_outs", defense)
    if pitcher_key:
        earned_match = [line for line in earned_candidates if line["player_key"] == pitcher_key]
        outs_match = [line for line in out_candidates if line["player_key"] == pitcher_key]
    else:
        earned_match, outs_match = [], []
    earned = _best(earned_match or earned_candidates)
    outs = _best(outs_match or out_candidates)
    if earned and outs:
        bullpen_innings = max(0.0, 9.0 - outs["v2_mean"] / 3.0)
        bullpen_runs = bullpen_innings * MLB_BULLPEN_RUNS_PER_INNING
        estimators.append({
            "name": "starter_er_plus_bullpen", "value": earned["v2_mean"] + bullpen_runs,
            "weight": min(1.0, min(earned["book_count"], outs["book_count"]) / 3.0),
            "inputs": 2, "starter": earned["player"], "starter_er": earned["v2_mean"],
            "starter_outs": outs["v2_mean"], "bullpen_innings": bullpen_innings,
            "bullpen_runs": bullpen_runs,
        })
    return estimators


def _combine_estimators(estimators):
    values = [(item["value"], item["weight"]) for item in estimators if item["weight"] > 0]
    if not values:
        return {}
    combinations = {
        "simple_median": statistics.median(value for value, _ in values),
        "quality_weighted_median": _weighted_median(values),
        "robust_quality_weighted_mean": _robust_weighted_mean(values),
    }
    combinations["consensus_candidate"] = statistics.median(combinations.values())
    return {name: min(15.0, max(0.0, value)) for name, value in combinations.items()}


def _mlb_projection(v1, lines, context):
    away, home = v1["away_team"], v1["home_team"]
    components = {
        away: _mlb_team_estimators(lines, away, home, away, home, context),
        home: _mlb_team_estimators(lines, home, away, away, home, context),
    }
    combined = {team: _combine_estimators(components[team]) for team in (away, home)}
    common = set(combined[away]) & set(combined[home])
    if not common:
        return {"status": "SHADOW_UNAVAILABLE", "reason": "missing_run_estimators"}
    variants = {}
    for method in sorted(common):
        away_mean, home_mean = combined[away][method], combined[home][method]
        variants[method] = {
            "away_mean": round(away_mean, 4), "home_mean": round(home_mean, 4),
            "away_score": _display_score(away_mean), "home_score": _display_score(home_mean),
        }
    confidence = v1.get("confidence") if v1.get("confidence") in {"HIGH", "MODERATE"} else "MODERATE"
    return {
        "status": "SHADOW_AVAILABLE", "confidence": confidence,
        "baseline_variant": "consensus_candidate", "variants": variants,
        "components": components,
        "assumptions": {
            "rbi_to_runs": MLB_RBI_TO_RUNS,
            "linear_weights": MLB_LINEAR_WEIGHTS,
            "bullpen_runs_per_inning": MLB_BULLPEN_RUNS_PER_INNING,
        },
    }


def project_event_v2(sport, v1_projection, canonical_lines, context=None, now=None):
    """Create an auditable shadow projection from the same canonical v1 lines."""
    base = {
        "shadow": True,
        "customer_facing": False,
        "sport": sport,
        "event_id": v1_projection.get("event_id"),
        "away_team": v1_projection.get("away_team"),
        "home_team": v1_projection.get("home_team"),
        "commence_time": v1_projection.get("commence_time"),
        "generated_at": (now.isoformat() if now else v1_projection.get("generated_at")),
        "model_version": V2_MODEL_VERSION.get(sport, f"prop_projection_{sport}_v2_shadow_1"),
        "source_v1_model_version": v1_projection.get("model_version"),
        "source_v1_audit_hash": v1_projection.get("audit_hash"),
        "v1_benchmark": {
            "status": v1_projection.get("status"), "confidence": v1_projection.get("confidence"),
            "away_score": v1_projection.get("away_score"), "home_score": v1_projection.get("home_score"),
            "away_mean": v1_projection.get("away_mean"), "home_mean": v1_projection.get("home_mean"),
        },
    }
    if not canonical_lines:
        return {**base, "status": "SHADOW_UNAVAILABLE", "reason": "no_canonical_lines"}
    lines = convert_lines(canonical_lines)
    conversion_audit = {
        "line_count": len(lines),
        "distribution_counts": dict(Counter(line["v2_kind"] for line in lines)),
        "minimum_mean_bounds_enforced": True,
    }
    if sport == "nfl":
        result = _nfl_projection(sport, v1_projection, lines)
    elif sport == "mlb":
        result = _mlb_projection(v1_projection, lines, context or {})
    else:
        result = {"status": "SHADOW_UNAVAILABLE", "reason": "sport_pending_v2_baseline"}
    return {**base, **result, "conversion_audit": conversion_audit}
