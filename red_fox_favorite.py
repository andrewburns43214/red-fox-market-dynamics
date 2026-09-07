"""Versioned Red Fox Favorite qualification layered over published Market Reads.

This module intentionally does not classify Market Reads or alter their rank.  It
consumes the two-side evidence already produced by :mod:`anomaly_board` and adds
an independent, binary customer designation plus a prospective audit ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


@dataclass(frozen=True)
class FavoriteConfig:
    version: str = "red_fox_favorite_v1"
    spread_sports: frozenset[str] = frozenset({"nfl", "ncaaf", "cfb", "nba", "ncaab", "cbb"})
    primary_moneyline_sports: frozenset[str] = frozenset({"mlb", "nhl", "ufc"})
    secondary_moneyline_sports: frozenset[str] = frozenset({"nfl", "ncaaf", "cfb", "nba", "ncaab", "cbb"})
    spread_min: float = 1.0
    spread_max: float = 8.0
    secondary_moneyline_spread_max: float = 4.0
    moneyline_min: int = -165
    moneyline_max: int = 125
    low_support_max: float = 45.0
    moderate_support_min: float = 46.0
    moderate_support_max: float = 60.0
    minimum_observations: int = 3
    meaningful_spread_move: float = 0.5
    meaningful_moneyline_price_move: float = 2.5


CONFIG = FavoriteConfig()
FAVORITE_COLUMNS = [
    "red_fox_favorite", "favorite_side", "favorite_pathway", "favorite_rule_version",
    "favorite_first_qualified_at", "favorite_final_qualified_at", "favorite_state",
    "favorite_final_market_read", "favorite_final_market_rank",
    "favorite_supporting_evidence", "favorite_whipsaw_state",
    "favorite_cross_market_state", "favorite_snapshot_id", "favorite_reason",
]
TRACKING_COLUMNS = [
    "recorded_at", "sport", "game_id", "game", "market_display", "favorite_state",
    "favorite_side", "favorite_pathway", "favorite_rule_version", "bets_pct", "money_pct",
    "first_qualified_at", "final_qualified_at", "open_line", "first_qualified_line", "current_line",
    "final_pregame_line", "final_pregame_market_read", "final_pregame_rank", "board_rank", "reaction", "path",
    "whipsaw_state", "cross_market_state", "supporting_evidence", "snapshot_id",
    "disappearance_reason", "closing_line", "clv", "result",
]


def apply_red_fox_favorites(board: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Add deterministic Favorite fields without changing row order or rank."""
    if board is None:
        return pd.DataFrame(columns=FAVORITE_COLUMNS)
    result = board.copy()
    for column in FAVORITE_COLUMNS:
        result[column] = ""
    result["red_fox_favorite"] = "false"
    result["favorite_rule_version"] = CONFIG.version
    result["favorite_state"] = "not_qualified"
    if result.empty:
        return result

    captured_at = _timestamp(as_of)
    groups = {
        (str(sport).lower(), str(game_id)): group
        for (sport, game_id), group in result.groupby(["sport", "game_id"], dropna=False, sort=False)
    }
    opener_blocked_games = set()
    for index, row in result.iterrows():
        decision = _qualify_market(row, groups.get((str(row.get("sport", "")).lower(), str(row.get("game_id", "")))))
        if not decision:
            continue
        current_mismatch = _truthy(row.get("cross_market_mismatch"))
        opener_mismatch = _truthy(row.get("cross_market_opener_mismatch_verified"))
        if current_mismatch or opener_mismatch:
            result.at[index, "favorite_cross_market_state"] = "mismatch"
            result.at[index, "favorite_reason"] = "Confirmed Cross-Market Mismatch withheld Favorite qualification."
            if opener_mismatch and not current_mismatch:
                opener_blocked_games.add((str(row.get("sport", "")).lower(), str(row.get("game_id", ""))))
            continue
        snapshot_id = _snapshot_id(row, captured_at)
        result.at[index, "red_fox_favorite"] = "true"
        result.at[index, "favorite_side"] = decision["side"]
        result.at[index, "favorite_pathway"] = decision["pathway"]
        result.at[index, "favorite_rule_version"] = CONFIG.version
        result.at[index, "favorite_first_qualified_at"] = captured_at
        result.at[index, "favorite_final_qualified_at"] = captured_at
        result.at[index, "favorite_state"] = "qualified"
        result.at[index, "favorite_final_market_read"] = decision["market_read"]
        result.at[index, "favorite_final_market_rank"] = row.get("board_rank", "")
        result.at[index, "favorite_supporting_evidence"] = json.dumps(decision["evidence"], separators=(",", ":"))
        result.at[index, "favorite_whipsaw_state"] = decision["whipsaw_state"]
        result.at[index, "favorite_cross_market_state"] = decision["cross_market_state"]
        result.at[index, "favorite_snapshot_id"] = snapshot_id
        result.at[index, "favorite_reason"] = decision["reason"]
    for sport, game_id in opener_blocked_games:
        pair_mask = (
            result["sport"].astype(str).str.lower().eq(sport)
            & result["game_id"].astype(str).eq(game_id)
            & result["market_display"].astype(str).str.upper().isin(["SPREAD", "MONEYLINE"])
        )
        explanation = next(
            (str(value) for value in result.loc[pair_mask, "cross_market_opener_mismatch_explanation"] if str(value).strip()),
            "The synchronized opening Spread and Moneyline implied different favorites.",
        )
        result.loc[pair_mask, "cross_market_mismatch"] = "true"
        result.loc[pair_mask, "cross_market_mismatch_state"] = "verified_opener"
        result.loc[pair_mask, "cross_market_mismatch_explanation"] = explanation
        if "market_rationale" in result.columns:
            for pair_index in result.index[pair_mask]:
                existing = str(result.at[pair_index, "market_rationale"] or "").strip()
                if explanation not in existing:
                    result.at[pair_index, "market_rationale"] = " ".join(part for part in (existing, explanation) if part)
    return result


def update_favorite_tracking(board: pd.DataFrame, data_dir: Path, as_of=None) -> pd.DataFrame:
    """Persist every prospective snapshot and carry first qualification forward."""
    current = board.copy()
    at = _timestamp(as_of)
    path = Path(data_dir) / "red_fox_favorite_tracking.csv"
    history = _read_tracking(path)
    keys = ["sport", "game_id", "market_display"]
    previous = {}
    if not history.empty:
        ordered = history.sort_values("recorded_at", kind="mergesort")
        previous = {
            tuple(str(row.get(column, "")) for column in keys): row
            for _, row in ordered.iterrows()
        }

    records = []
    current_keys = set()
    for index, row in current.iterrows():
        key = tuple(str(row.get(column, "")) for column in keys)
        current_keys.add(key)
        prior = previous.get(key)
        favorite = _truthy(row.get("red_fox_favorite"))
        if favorite and prior is not None:
            prior_first = str(prior.get("first_qualified_line", "")).strip()
            prior_at = str(prior.get("recorded_at", "")).strip()
            first_at = _first_qualification_time(history, key) or prior_at or at
            current.at[index, "favorite_first_qualified_at"] = first_at
            first_line = prior_first or str(row.get("current_line", ""))
        else:
            first_line = str(row.get("current_line", "")) if favorite else ""
        same_snapshot = (
            favorite and prior is not None
            and str(prior.get("favorite_state", "")) == "qualified"
            and str(prior.get("snapshot_id", "")) == str(row.get("favorite_snapshot_id", ""))
        )
        if favorite and not same_snapshot:
            records.append(_tracking_record(current.loc[index], at, first_line))
        elif prior is not None and str(prior.get("favorite_state", "")) == "qualified":
            record = {column: str(prior.get(column, "")) for column in TRACKING_COLUMNS}
            record.update(
                recorded_at=at,
                favorite_state="not_qualified",
                disappearance_reason="current qualification gates no longer satisfied",
            )
            records.append(record)

    # A previously qualifying market can leave the board because it started or
    # ceased to meet a gate. Record that transition without rewriting its frozen
    # Live & Recent row.
    for key, prior in previous.items():
        if key in current_keys or str(prior.get("favorite_state", "")) != "qualified":
            continue
        record = {column: str(prior.get(column, "")) for column in TRACKING_COLUMNS}
        record.update(recorded_at=at, favorite_state="not_qualified", disappearance_reason="no longer published or eligible")
        records.append(record)

    if records:
        new_records = pd.DataFrame(records)
        updated = new_records if history.empty else pd.concat([history, new_records], ignore_index=True, sort=False)
        updated = updated.reindex(columns=TRACKING_COLUMNS)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + ".tmp")
        updated.to_csv(temporary, index=False)
        temporary.replace(path)
    return current


def _qualify_market(row: pd.Series, game_rows: pd.DataFrame | None) -> dict | None:
    sport = str(row.get("sport", "")).lower()
    market = str(row.get("market_display", "")).upper()
    if market == "TOTAL":
        return None
    if market == "SPREAD" and sport not in CONFIG.spread_sports:
        return None
    if market == "MONEYLINE" and sport not in CONFIG.primary_moneyline_sports | CONFIG.secondary_moneyline_sports:
        return None
    sides = _sides(row)
    if len(sides) != 2:
        return None

    candidates = []
    for side in sides:
        other = next((item for item in sides if item is not side), None)
        if other is None or not _base_quality(side):
            continue
        value = _line_value(side.get("current_line"), market)
        if value is None or not _number_is_eligible(sport, market, value, side, game_rows):
            continue
        pathway = _pathway(side, other)
        if not pathway:
            continue
        cross_state = "not_applicable"
        if market == "SPREAD":
            cross_state = _corresponding_moneyline_state(side, game_rows)
            if cross_state == "contradiction":
                continue
        candidates.append((side, pathway, cross_state))
    if not candidates:
        return None
    priority = {"low_support_contrarian": 0, "low_support_freeze": 1, "moderate_support_follow": 2}
    side, pathway, cross_state = min(candidates, key=lambda item: priority[item[1]])
    context = _parts(side.get("context_chips"))
    whipsaw_state = "recovered" if _truthy(side.get("whipsaw_recovered")) or "Whipsaw Recovered" in context else (
        "partial_retrace_intact" if str(side.get("path", "")) == "Whipsaw" else "none"
    )
    evidence = {
        "reaction": str(side.get("reaction", "")),
        "path": str(side.get("path", "")),
        "bets_pct": _number(side.get("bets_pct")),
        "money_pct": _number(side.get("money_pct")),
        "open_line": str(side.get("open_line", "")),
        "current_line": str(side.get("current_line", "")),
        "observation_count": int(_number(side.get("observation_count")) or 0),
        "cross_market": cross_state,
    }
    reason = {
        "low_support_contrarian": "Lower-supported side received a meaningful, intact move toward it.",
        "low_support_freeze": "Lower-supported side is protected by the engine's actionable, persistent resistance state.",
        "moderate_support_follow": "Moderately supported side received a meaningful, intact confirming move.",
    }[pathway]
    return {
        "side": str(side.get("flagged_side", "")), "pathway": pathway,
        "cross_market_state": cross_state, "whipsaw_state": whipsaw_state,
        "evidence": evidence, "reason": reason,
        "market_read": str(side.get("anomaly_chips") or side.get("reaction", "")),
    }


def _pathway(side: dict, other: dict) -> str:
    bets, money = _number(side.get("bets_pct")), _number(side.get("money_pct"))
    other_bets, other_money = _number(other.get("bets_pct")), _number(other.get("money_pct"))
    if None in {bets, money, other_bets, other_money}:
        return ""
    low = bets <= CONFIG.low_support_max and money <= CONFIG.low_support_max and bets < other_bets and money < other_money
    toward = str(side.get("response_direction", "")).upper() == "TOWARD"
    reaction = str(side.get("reaction", ""))
    if low and reaction == "Contrarian" and toward and _truthy(side.get("kpi_eligible")):
        return "low_support_contrarian"
    if low and _is_favorite_resistance(side, other):
        return "low_support_freeze"
    moderate = (
        CONFIG.moderate_support_min <= bets <= CONFIG.moderate_support_max
        and CONFIG.moderate_support_min <= money <= CONFIG.moderate_support_max
    )
    if moderate and toward and _has_meaningful_move(side):
        return "moderate_support_follow"
    return ""


def _is_favorite_resistance(side: dict, pressure: dict) -> bool:
    """Evaluate Favorite Path B without inheriting generic fade safeguards.

    The Market Read engine places ``Freeze`` on the high-support pressure side
    and ``Resistance Side`` on the protected, low-support counterpart.  Its
    generic fade candidate additionally blocks key numbers and requires 80%
    tickets; neither safeguard belongs to the Favorite contract.
    """
    if str(pressure.get("reaction", "")) != "Freeze" or not _base_quality(pressure):
        return False
    pressure_bets = _number(pressure.get("bets_pct"))
    pressure_money = _number(pressure.get("money_pct"))
    if pressure_bets is None or pressure_money is None:
        return False
    pressure_context = _parts(pressure.get("context_chips"))
    substantial_pressure = (
        "Public Pressure" in pressure_context
        or (pressure_bets >= 70 and pressure_money >= 55)
    )
    if not substantial_pressure:
        return False
    # Freeze means concentrated pressure failed to earn a meaningful favorable
    # response.  The paired side must still be held/protected, not moving
    # materially against the proposed Favorite.
    if str(pressure.get("response_direction", "")).upper() not in {"AGAINST", "LIMITED"}:
        return False
    if str(side.get("response_direction", "")).upper() not in {"TOWARD", "LIMITED"}:
        return False
    pressure_role = str(pressure.get("evidence_role", "")).strip()
    resistance_role = str(side.get("evidence_role", "")).strip()
    if pressure_role and pressure_role != "Pressure Side":
        return False
    if resistance_role and resistance_role != "Resistance Side":
        return False
    return True


def _base_quality(side: dict) -> bool:
    if str(side.get("data_badge", "")) != "Clean":
        return False
    if (_number(side.get("observation_count")) or 0) < CONFIG.minimum_observations:
        return False
    context = _parts(side.get("context_chips"))
    if {"Market Lag", "Feed Risk", "Split Risk", "Split Cap"} & context:
        return False
    if _truthy(side.get("active_worsening_reversal")):
        return False
    # A Whipsaw is allowed when meaningful favorable movement remains at the
    # current snapshot. Full/through-open reversals no longer report TOWARD and
    # therefore cannot pass Paths A/C; actionable Freeze already rejects them.
    return True


def _number_is_eligible(sport: str, market: str, value: float, side: dict, game_rows: pd.DataFrame | None) -> bool:
    if market == "SPREAD":
        return CONFIG.spread_min <= abs(value) <= CONFIG.spread_max
    if not CONFIG.moneyline_min <= value <= CONFIG.moneyline_max:
        return False
    if sport in CONFIG.primary_moneyline_sports:
        return True
    spread = _matching_market_side(game_rows, "SPREAD", side.get("flagged_side"))
    spread_value = _line_value(spread.get("current_line"), "SPREAD") if spread else None
    return spread_value is not None and abs(spread_value) <= CONFIG.secondary_moneyline_spread_max


def _corresponding_moneyline_state(spread_side: dict, game_rows: pd.DataFrame | None) -> str:
    moneyline = _matching_market_side(game_rows, "MONEYLINE", spread_side.get("flagged_side"))
    if not moneyline:
        return "unavailable"
    direction = str(moneyline.get("response_direction", "")).upper()
    meaningful = (_number(moneyline.get("price_move_pct")) or 0) >= CONFIG.meaningful_moneyline_price_move
    if meaningful and direction == "AGAINST":
        return "contradiction"
    if meaningful and direction == "TOWARD":
        return "confirmation"
    return "neutral"


def _matching_market_side(game_rows: pd.DataFrame | None, market: str, side_name: object) -> dict | None:
    if game_rows is None or game_rows.empty:
        return None
    target = _side_identity(side_name)
    rows = game_rows[game_rows["market_display"].astype(str).str.upper().eq(market)]
    for _, row in rows.iterrows():
        for side in _sides(row):
            if _side_identity(side.get("flagged_side")) == target:
                return side
    return None


def _has_meaningful_move(side: dict) -> bool:
    market = "MONEYLINE" if not re.search(r"\s[+-]\d+(?:\.\d+)?(?:\s|$)", str(side.get("flagged_side", ""))) else "SPREAD"
    if market == "MONEYLINE":
        return (_number(side.get("price_move_pct")) or 0) >= CONFIG.meaningful_moneyline_price_move
    return (
        (_number(side.get("line_move_abs")) or 0) >= CONFIG.meaningful_spread_move
        or (_number(side.get("price_move_pct")) or 0) >= CONFIG.meaningful_moneyline_price_move
    )


def _sides(row: pd.Series | dict) -> list[dict]:
    try:
        value = json.loads(str(row.get("market_sides", "[]")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _line_value(value: object, market: str) -> float | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if market == "MONEYLINE":
        match = re.search(r"(?<!\d)([+-]?\d{3,4})(?!\d)", text)
    else:
        match = re.search(r"(?:^|\s)([+-]?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _side_identity(value: object) -> str:
    text = re.sub(r"\s[+-]\d+(?:\.\d+)?(?:\s.*)?$", "", str(value or "").strip())
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _parts(value: object) -> set[str]:
    return {part.strip() for part in str(value or "").split("|") if part.strip()}


def _number(value: object) -> float | None:
    try:
        number = float(value)
        return number if pd.notna(number) else None
    except (TypeError, ValueError):
        return None


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _timestamp(value: object) -> str:
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _snapshot_id(row: pd.Series, at: str) -> str:
    source = "|".join(str(row.get(column, "")) for column in ("sport", "game_id", "market_display", "current_line")) + "|" + at
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]


def _read_tracking(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() and path.stat().st_size else pd.DataFrame(columns=TRACKING_COLUMNS)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=TRACKING_COLUMNS)


def _first_qualification_time(history: pd.DataFrame, key: tuple[str, str, str]) -> str:
    if history.empty:
        return ""
    mask = pd.Series(True, index=history.index)
    for column, value in zip(("sport", "game_id", "market_display"), key):
        mask &= history[column].astype(str).eq(value)
    qualified = history[mask & history["favorite_state"].eq("qualified")]
    return str(qualified.iloc[0]["recorded_at"]) if not qualified.empty else ""


def _tracking_record(row: pd.Series, at: str, first_line: str) -> dict:
    try:
        evidence = json.loads(str(row.get("favorite_supporting_evidence", "{}")))
    except (TypeError, ValueError, json.JSONDecodeError):
        evidence = {}
    return {
        "recorded_at": at, "sport": row.get("sport", ""), "game_id": row.get("game_id", ""),
        "game": row.get("game", ""), "market_display": row.get("market_display", ""),
        "favorite_state": "qualified", "favorite_side": row.get("favorite_side", ""),
        "favorite_pathway": row.get("favorite_pathway", ""), "favorite_rule_version": CONFIG.version,
        "bets_pct": evidence.get("bets_pct", ""), "money_pct": evidence.get("money_pct", ""),
        "first_qualified_at": row.get("favorite_first_qualified_at", at),
        "final_qualified_at": row.get("favorite_final_qualified_at", at),
        "open_line": evidence.get("open_line", ""), "first_qualified_line": first_line,
        "current_line": evidence.get("current_line", ""), "board_rank": row.get("board_rank", ""),
        "final_pregame_line": evidence.get("current_line", ""),
        "final_pregame_market_read": row.get("favorite_final_market_read", ""),
        "final_pregame_rank": row.get("favorite_final_market_rank", row.get("board_rank", "")),
        "reaction": evidence.get("reaction", ""), "path": evidence.get("path", ""),
        "whipsaw_state": row.get("favorite_whipsaw_state", ""),
        "cross_market_state": row.get("favorite_cross_market_state", ""),
        "supporting_evidence": row.get("favorite_supporting_evidence", ""),
        "snapshot_id": row.get("favorite_snapshot_id", ""), "disappearance_reason": "",
        "closing_line": "", "clv": "", "result": "",
    }
