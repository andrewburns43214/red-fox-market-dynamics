"""Narrow Spread/Moneyline coherence checks for the customer board.

This module is deliberately separate from market-read scoring.  It annotates a
paired market only when complete, synchronized DraftKings observations disagree
about which team is favored and that disagreement is persistent.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

import pandas as pd


@dataclass(frozen=True)
class CrossMarketIntegrityConfig:
    max_pair_minutes: float = 15.0
    minimum_spread: float = 1.5
    dog_probability_floor: float = 0.53
    favorite_probability_ceiling: float = 0.47
    minimum_observations: int = 3
    minimum_duration_minutes: float = 20.0
    maximum_streak_gap_minutes: float = 20.0


CONFIG = CrossMarketIntegrityConfig()
CROSS_MARKET_COLUMNS = [
    "cross_market_mismatch",
    "cross_market_mismatch_state",
    "cross_market_mismatch_explanation",
    "cross_market_mismatch_team",
    "cross_market_no_vig_probability",
    "cross_market_observation_count",
    "cross_market_duration_minutes",
    "cross_market_spread_observed_at",
    "cross_market_moneyline_observed_at",
    "cross_market_spread_current",
    "cross_market_moneyline_current",
    "cross_market_opener_mismatch_verified",
    "cross_market_opener_mismatch_explanation",
]


def apply_cross_market_integrity(board: pd.DataFrame, history: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Annotate paired markets without changing order, reads, or rank."""
    if board is None:
        return pd.DataFrame(columns=CROSS_MARKET_COLUMNS)
    result = board.copy()
    for column in CROSS_MARKET_COLUMNS:
        result[column] = ""
    result["cross_market_mismatch"] = "false"
    result["cross_market_opener_mismatch_verified"] = "false"
    if result.empty or history is None or history.empty:
        return result

    prepared = _prepare_history(history)
    if prepared.empty:
        return result
    captured_at = _utc_timestamp(as_of)
    for (sport, game_id), indexes in result.groupby(["sport", "game_id"], dropna=False, sort=False).groups.items():
        game_rows = result.loc[indexes]
        markets = set(game_rows["market_display"].astype(str).str.upper())
        if not {"SPREAD", "MONEYLINE"}.issubset(markets):
            continue
        if not _board_pair_reliable(game_rows):
            continue
        game_history = prepared[
            prepared["sport"].astype(str).str.lower().eq(str(sport).lower())
            & prepared["game_id"].astype(str).eq(str(game_id))
        ]
        evaluation = evaluate_cross_market_history(game_history, as_of=captured_at)
        pair_mask = result.index.isin(indexes) & result["market_display"].astype(str).str.upper().isin(["SPREAD", "MONEYLINE"])
        if evaluation["opener_verified"]:
            result.loc[pair_mask, "cross_market_opener_mismatch_verified"] = "true"
            result.loc[pair_mask, "cross_market_opener_mismatch_explanation"] = evaluation["opener_explanation"]
        if not evaluation["confirmed"]:
            continue
        values = {
            "cross_market_mismatch": "true",
            "cross_market_mismatch_state": "confirmed_current",
            "cross_market_mismatch_explanation": evaluation["explanation"],
            "cross_market_mismatch_team": evaluation["team"],
            "cross_market_no_vig_probability": _format_probability(evaluation["probability"]),
            "cross_market_observation_count": str(evaluation["observation_count"]),
            "cross_market_duration_minutes": _format_number(evaluation["duration_minutes"]),
            "cross_market_spread_observed_at": evaluation["spread_observed_at"],
            "cross_market_moneyline_observed_at": evaluation["moneyline_observed_at"],
            "cross_market_spread_current": evaluation["spread_current"],
            "cross_market_moneyline_current": evaluation["moneyline_current"],
        }
        for column, value in values.items():
            result.loc[pair_mask, column] = value
        if "market_rationale" in result.columns:
            for index in result.index[pair_mask]:
                existing = str(result.at[index, "market_rationale"] or "").strip()
                explanation = evaluation["explanation"]
                if explanation and explanation not in existing:
                    result.at[index, "market_rationale"] = " ".join(part for part in (existing, explanation) if part)
    return result


def evaluate_cross_market_history(history: pd.DataFrame, as_of=None) -> dict:
    """Return a deterministic current and opener integrity evaluation."""
    empty = {
        "confirmed": False, "explanation": "", "team": "", "probability": None,
        "observation_count": 0, "duration_minutes": 0.0,
        "spread_observed_at": "", "moneyline_observed_at": "",
        "spread_current": "", "moneyline_current": "",
        "opener_verified": False, "opener_explanation": "",
    }
    if history is None or history.empty:
        return empty
    prepared = _prepare_history(history)
    spread = _market_observations(prepared, "SPREAD")
    moneyline = _market_observations(prepared, "MONEYLINE")
    pairs = _pair_observations(spread, moneyline)
    if not pairs:
        return empty

    current = _conflict_streak(pairs, use_open=False)
    latest = pairs[-1]
    reference = _utc_timestamp(as_of) if as_of is not None else max(latest["spread_time"], latest["moneyline_time"])
    current_fresh = (
        reference - latest["spread_time"] <= pd.Timedelta(minutes=CONFIG.max_pair_minutes)
        and reference - latest["moneyline_time"] <= pd.Timedelta(minutes=CONFIG.max_pair_minutes)
    )
    confirmed = bool(current["conflict"] and current_fresh and (
        current["count"] >= CONFIG.minimum_observations
        or current["duration"] >= CONFIG.minimum_duration_minutes
    ))

    first_spread, first_moneyline = spread[0], moneyline[0]
    contemporaneous_openers = abs(first_spread["time"] - first_moneyline["time"]) <= pd.Timedelta(minutes=CONFIG.max_pair_minutes)
    immutable_openers = _openers_are_immutable(spread) and _openers_are_immutable(moneyline)
    opener = _conflict_streak(pairs, use_open=True)
    opener_verified = bool(
        contemporaneous_openers and immutable_openers and opener["conflict"]
        and (opener["count"] >= CONFIG.minimum_observations or opener["duration"] >= CONFIG.minimum_duration_minutes)
    )

    result = dict(empty)
    result.update(
        confirmed=confirmed,
        explanation=current["explanation"] if confirmed else "",
        team=current["team"] if confirmed else "",
        probability=current["probability"] if confirmed else None,
        observation_count=current["count"] if confirmed else 0,
        duration_minutes=current["duration"] if confirmed else 0.0,
        spread_observed_at=latest["spread_time"].isoformat() if confirmed else "",
        moneyline_observed_at=latest["moneyline_time"].isoformat() if confirmed else "",
        spread_current=_display_map(latest["spread_current"]) if confirmed else "",
        moneyline_current=_display_map(latest["moneyline_current"]) if confirmed else "",
        opener_verified=opener_verified,
        opener_explanation=opener["explanation"] if opener_verified else "",
    )
    return result


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    required = {"sport", "game_id", "side", "current_line", "open_line", "timestamp"}
    if history is None or history.empty or not required.issubset(history.columns):
        return pd.DataFrame()
    work = history.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work[work["timestamp"].notna()].copy()
    if "market_display" not in work.columns:
        work["market_display"] = work.apply(_infer_market, axis=1)
    work["market_display"] = work["market_display"].astype(str).str.upper()
    return work[work["market_display"].isin(["SPREAD", "MONEYLINE"])]


def _market_observations(history: pd.DataFrame, market: str) -> list[dict]:
    rows = history[history["market_display"].eq(market)]
    observations = []
    for timestamp, captured in rows.groupby("timestamp", sort=True):
        current, opening, displays = {}, {}, {}
        reliable = True
        for _, row in captured.iterrows():
            team = _team_identity(row.get("side"))
            if not team or team in current:
                reliable = False
                break
            current_value = _spread_value(row.get("current_line"), row.get("side")) if market == "SPREAD" else _odds(row.get("current_line"))
            open_value = _spread_value(row.get("open_line"), row.get("side")) if market == "SPREAD" else _odds(row.get("open_line"))
            if current_value is None or open_value is None or _unreliable_text(row.get("current_line")) or _unreliable_text(row.get("open_line")):
                reliable = False
                break
            current[team], opening[team], displays[team] = current_value, open_value, _team_label(row.get("side"))
        if reliable and len(current) == 2:
            observations.append({"time": timestamp, "current": current, "open": opening, "labels": displays})
    return observations


def _pair_observations(spread: list[dict], moneyline: list[dict]) -> list[dict]:
    pairs, unused = [], set(range(len(moneyline)))
    for spread_observation in spread:
        candidates = [
            index for index in unused
            if set(spread_observation["current"]) == set(moneyline[index]["current"])
            and abs(spread_observation["time"] - moneyline[index]["time"]) <= pd.Timedelta(minutes=CONFIG.max_pair_minutes)
        ]
        if not candidates:
            continue
        index = min(candidates, key=lambda item: abs(spread_observation["time"] - moneyline[item]["time"]))
        unused.remove(index)
        ml_observation = moneyline[index]
        pairs.append({
            "time": max(spread_observation["time"], ml_observation["time"]),
            "spread_time": spread_observation["time"], "moneyline_time": ml_observation["time"],
            "spread_current": spread_observation["current"], "moneyline_current": ml_observation["current"],
            "spread_open": spread_observation["open"], "moneyline_open": ml_observation["open"],
            "labels": spread_observation["labels"],
        })
    return sorted(pairs, key=lambda pair: pair["time"])


def _conflict_streak(pairs: list[dict], use_open: bool) -> dict:
    evaluations = [_evaluate_pair(pair, use_open=use_open) for pair in pairs]
    latest = evaluations[-1]
    if not latest["conflict"]:
        return {**latest, "count": 0, "duration": 0.0}
    streak = [(pairs[-1], latest)]
    for pair, evaluation in reversed(list(zip(pairs[:-1], evaluations[:-1]))):
        newer_pair, newer_evaluation = streak[-1]
        gap = (newer_pair["time"] - pair["time"]).total_seconds() / 60.0
        same_story = evaluation["conflict"] and evaluation["team"] == newer_evaluation["team"]
        if not same_story or gap > CONFIG.maximum_streak_gap_minutes:
            break
        streak.append((pair, evaluation))
    duration = (streak[0][0]["time"] - streak[-1][0]["time"]).total_seconds() / 60.0
    return {**latest, "count": len(streak), "duration": max(0.0, duration)}


def _evaluate_pair(pair: dict, use_open: bool) -> dict:
    spread = pair["spread_open" if use_open else "spread_current"]
    moneyline = pair["moneyline_open" if use_open else "moneyline_current"]
    probabilities = _no_vig_probabilities(moneyline)
    if not probabilities:
        return {"conflict": False, "team": "", "probability": None, "explanation": ""}
    for team, spread_value in spread.items():
        probability = probabilities.get(team)
        if probability is None or abs(spread_value) < CONFIG.minimum_spread:
            continue
        conflict = (
            spread_value > 0 and probability >= CONFIG.dog_probability_floor
        ) or (
            spread_value < 0 and probability <= CONFIG.favorite_probability_ceiling
        )
        if not conflict:
            continue
        spread_favorite = next((key for key, value in spread.items() if value < 0), "")
        ml_favorite = max(probabilities, key=probabilities.get)
        labels = pair["labels"]
        explanation = (
            f"The synchronized Spread prices {labels.get(spread_favorite, spread_favorite)} as the favorite "
            f"while the Moneyline prices {labels.get(ml_favorite, ml_favorite)} as the favorite."
        )
        return {"conflict": True, "team": labels.get(team, team), "probability": probability, "explanation": explanation}
    return {"conflict": False, "team": "", "probability": None, "explanation": ""}


def _no_vig_probabilities(odds_by_team: dict[str, float]) -> dict[str, float]:
    if len(odds_by_team) != 2:
        return {}
    raw = {team: _implied_probability(odds) for team, odds in odds_by_team.items()}
    if any(value is None for value in raw.values()):
        return {}
    total = sum(raw.values())
    return {team: value / total for team, value in raw.items()} if total > 0 else {}


def _implied_probability(odds: float) -> float | None:
    if odds == 0:
        return None
    return abs(odds) / (abs(odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def _board_pair_reliable(game_rows: pd.DataFrame) -> bool:
    paired = game_rows[game_rows["market_display"].astype(str).str.upper().isin(["SPREAD", "MONEYLINE"])]
    if set(paired["market_display"].astype(str).str.upper()) != {"SPREAD", "MONEYLINE"}:
        return False
    for _, row in paired.iterrows():
        try:
            sides = json.loads(str(row.get("market_sides", "[]")))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(sides, list) or len(sides) != 2:
            return False
        for side in sides:
            if str(side.get("data_badge", "")) != "Clean":
                return False
            context = {part.strip() for part in str(side.get("context_chips", "")).split("|") if part.strip()}
            if {"Market Lag", "Feed Risk", "Split Risk", "Split Cap"} & context:
                return False
    return True


def _openers_are_immutable(observations: list[dict]) -> bool:
    if not observations:
        return False
    first = observations[0]["open"]
    return all(observation["open"] == first for observation in observations)


def _infer_market(row: pd.Series) -> str:
    side = str(row.get("side", ""))
    if re.match(r"^(over|under)\b", side, flags=re.IGNORECASE):
        return "TOTAL"
    if re.search(r"\s[+-]\d+(?:\.\d+)?(?:\s|$)", side):
        return "SPREAD"
    return "MONEYLINE"


def _team_identity(value: object) -> str:
    text = re.sub(r"\s[+-]\d+(?:\.\d+)?(?:\s.*)?$", "", str(value or "").strip())
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _team_label(value: object) -> str:
    return re.sub(r"\s[+-]\d+(?:\.\d+)?(?:\s.*)?$", "", str(value or "").strip())


def _spread_value(line: object, side: object) -> float | None:
    text = str(line or "")
    match = re.search(r"\s([+-]\d+(?:\.\d+)?)\s*@", text)
    if not match:
        match = re.search(r"\s([+-]\d+(?:\.\d+)?)(?:\s|$)", str(side or ""))
    return float(match.group(1)) if match else None


def _odds(line: object) -> float | None:
    match = re.search(r"@\s*([+-]\d{3,4})(?!\d)", str(line or ""))
    if not match:
        match = re.search(r"(?<!\d)([+-]\d{3,4})(?!\d)", str(line or ""))
    return float(match.group(1)) if match else None


def _unreliable_text(value: object) -> bool:
    return bool(re.search(r"suspend|offline|unavailable|tbd", str(value or ""), flags=re.IGNORECASE))


def _display_map(values: dict[str, float]) -> str:
    return json.dumps(values, separators=(",", ":"), sort_keys=True)


def _format_probability(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.2f}%"


def _format_number(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
