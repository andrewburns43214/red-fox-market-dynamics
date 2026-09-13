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
    version: str = "red_fox_favorite_v2"
    spread_sports: frozenset[str] = frozenset({"nfl", "ncaaf", "cfb", "nba", "ncaab", "cbb"})
    primary_moneyline_sports: frozenset[str] = frozenset({"mlb", "nhl", "ufc"})
    secondary_moneyline_sports: frozenset[str] = frozenset({"nfl", "ncaaf", "cfb", "nba", "ncaab", "cbb"})
    spread_min: float = 1.0
    spread_max: float = 8.0
    secondary_moneyline_spread_max: float = 4.0
    moneyline_min: int = -165
    moneyline_max: int = 125
    primary_moneyline_max: int = 120
    low_support_max: float = 45.0
    moderate_support_min: float = 46.0
    moderate_support_max: float = 60.0
    minimum_observations: int = 3
    meaningful_spread_move: float = 0.5
    meaningful_moneyline_price_move: float = 2.5
    resistance_min_bets: float = 80.0
    resistance_min_money: float = 60.0
    resistance_max_direction_changes: int = 4
    football_favorite_max_bets: float = 40.0
    football_favorite_max_money: float = 39.999
    football_favorite_max_spread_money: float = 49.999
    football_short_home_max_money: float = 35.0
    football_short_home_max_spread_money: float = 40.0
    football_favorite_max_direction_changes: int = 4
    freeze_disabled_sports: frozenset[str] = frozenset({"ufc"})


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
    _update_freeze_candidates(current, Path(data_dir), at)
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
        pathway = _pathway(side, other, sport, market)
        if not pathway:
            continue
        # Favorite is a narrower designation layered over the published
        # Market Read.  Never publish a Favorite when that same side is not
        # the market's authoritative confirmed supported side; otherwise the
        # red badge can appear with no corresponding green row (or on the
        # opposite row).  Do not infer support here from movement/context --
        # ``supported_side`` is resolved upstream by the Market Read engine.
        if not _matches_confirmed_supported_side(row, side):
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


def _pathway(side: dict, other: dict, sport: str, market: str) -> str:
    bets, money = _number(side.get("bets_pct")), _number(side.get("money_pct"))
    other_bets, other_money = _number(other.get("bets_pct")), _number(other.get("money_pct"))
    if None in {bets, money, other_bets, other_money}:
        return ""
    low = bets <= CONFIG.low_support_max and money <= CONFIG.low_support_max and bets < other_bets and money < other_money
    toward = str(side.get("response_direction", "")).upper() == "TOWARD"
    reaction = str(side.get("reaction", ""))
    if low and reaction == "Contrarian" and toward and _truthy(side.get("kpi_eligible")):
        return "low_support_contrarian"
    if low and _is_favorite_resistance(side, other, sport, market):
        return "low_support_freeze"
    moderate = (
        CONFIG.moderate_support_min <= bets <= CONFIG.moderate_support_max
        and CONFIG.moderate_support_min <= money <= CONFIG.moderate_support_max
    )
    if moderate and toward and _has_meaningful_move(side):
        return "moderate_support_follow"
    return ""


def _is_favorite_resistance(side: dict, pressure: dict, sport: str, market: str) -> bool:
    """Require a clean, persistent resistance state for Favorite Path B."""
    if sport in CONFIG.freeze_disabled_sports:
        return False
    if str(pressure.get("reaction", "")) != "Freeze" or not _base_quality(pressure):
        return False
    pressure_bets = _number(pressure.get("bets_pct"))
    pressure_money = _number(pressure.get("money_pct"))
    if pressure_bets is None or pressure_money is None:
        return False
    substantial_pressure = (
        pressure_bets >= CONFIG.resistance_min_bets
        and pressure_money >= CONFIG.resistance_min_money
        and _truthy(pressure.get("kpi_eligible"))
        and str(pressure.get("action_type", "")).upper() == "FADE CANDIDATE"
    )
    if not substantial_pressure:
        return False
    # A Favorite is stronger than a descriptive Freeze. Do not elevate a
    # market that merely wandered back toward its opener or repeatedly changed
    # direction inside a small price range.
    if _truthy(side.get("return_toward_open")) or _truthy(pressure.get("return_toward_open")):
        return False
    direction_changes = max(
        int(_number(side.get("line_dir_changes")) or 0),
        int(_number(pressure.get("line_dir_changes")) or 0),
    )
    if direction_changes > CONFIG.resistance_max_direction_changes:
        return False
    # Freeze means concentrated pressure failed to earn a meaningful favorable
    # response.  The paired side must still be held/protected, not moving
    # materially against the proposed Favorite.
    if str(pressure.get("response_direction", "")).upper() not in {"AGAINST", "LIMITED"}:
        return False
    if str(side.get("response_direction", "")).upper() not in {"TOWARD", "LIMITED"}:
        return False
    held = str(pressure.get("path", "")) == "Held" and str(side.get("path", "")) == "Held"
    adverse_move = str(pressure.get("response_direction", "")).upper() == "AGAINST" and (
        (_number(pressure.get("price_move_pct")) or 0) >= CONFIG.meaningful_moneyline_price_move
        or (
            market == "SPREAD"
            and (_number(pressure.get("line_move_abs")) or 0) >= CONFIG.meaningful_spread_move
        )
    )
    if not held and not adverse_move:
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
    maximum = CONFIG.primary_moneyline_max if sport in CONFIG.primary_moneyline_sports else CONFIG.moneyline_max
    if not CONFIG.moneyline_min <= value <= maximum:
        return False
    if sport in CONFIG.primary_moneyline_sports:
        return True
    spread = _matching_market_side(game_rows, "SPREAD", side.get("flagged_side"))
    spread_value = _line_value(spread.get("current_line"), "SPREAD") if spread else None
    if spread_value is None or abs(spread_value) > CONFIG.secondary_moneyline_spread_max:
        return False
    # In football and basketball, positive money is directional/cross-market
    # evidence rather than a separately published Favorite.  A team receiving
    # points likewise uses its independently qualified spread as the sole
    # Favorite expression, preventing two correlated KPI results for one
    # thesis.  Pick'em does not manufacture a positive-spread expression.
    if value > 0 or spread_value > 0:
        return False
    if sport in {"nfl", "ncaaf", "cfb"} and not _football_favorite_moneyline_gate(
        side, spread, spread_value
    ):
        return False
    return True


def _football_favorite_moneyline_gate(side: dict, spread: dict, spread_value: float) -> bool:
    """Apply the stronger final-state standard to negative-money football Favorites.

    This is deliberately scoped away from point-taking spread Favorites.  It
    prevents a broadly supported, unstable home favorite such as Texas from
    becoming a contrarian Favorite merely because its moneyline split is just
    under the legacy 45 percent ceiling.
    """
    bets = _number(side.get("bets_pct"))
    money = _number(side.get("money_pct"))
    spread_money = _number(spread.get("money_pct"))
    if bets is None or money is None or spread_money is None:
        return False
    if bets > CONFIG.football_favorite_max_bets or money > CONFIG.football_favorite_max_money:
        return False
    if spread_money > CONFIG.football_favorite_max_spread_money:
        return False
    if _truthy(side.get("return_toward_open")) or _truthy(spread.get("return_toward_open")):
        return False
    direction_changes = max(
        int(_number(side.get("line_dir_changes")) or 0),
        int(_number(spread.get("line_dir_changes")) or 0),
    )
    if direction_changes > CONFIG.football_favorite_max_direction_changes:
        return False
    # A true home favorite can be shorter than the key number three, but that
    # weaker market position must be accompanied by exceptional split support.
    if -3 < spread_value < 0:
        if money > CONFIG.football_short_home_max_money:
            return False
        if spread_money > CONFIG.football_short_home_max_spread_money:
            return False
    return True


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


def _matches_confirmed_supported_side(row: pd.Series, candidate: dict) -> bool:
    """Require one exact published-side match for the Favorite candidate."""
    supported_identity = _side_identity(row.get("supported_side", ""))
    candidate_identity = _side_identity(candidate.get("flagged_side", ""))
    if not supported_identity or supported_identity != candidate_identity:
        return False
    published_identities = [
        _side_identity(side.get("flagged_side", ""))
        for side in _sides(row)
        if isinstance(side, dict)
    ]
    return published_identities.count(supported_identity) == 1


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


def _update_freeze_candidates(current: pd.DataFrame, data_dir: Path, at: str) -> None:
    """Retain the last qualified full board row independently of board freshness.

    The public pregame board is intentionally transient.  This archive is the
    immutable handoff source used at kickoff when a qualified market vanishes
    from one scrape or crosses the board freshness boundary.
    """
    path = data_dir / "red_fox_favorite_freeze_candidates.csv"
    try:
        archived = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() and path.stat().st_size else pd.DataFrame()
    except (OSError, pd.errors.EmptyDataError):
        archived = pd.DataFrame()
    if current.empty:
        return
    keys = ["sport", "game_id", "market_display"]
    if not all(column in current.columns for column in keys):
        return
    current = current.copy()
    if "state_as_of_utc" not in current:
        current["state_as_of_utc"] = at
    current["candidate_recorded_at_utc"] = at
    if archived.empty:
        favorite_values = current.get("red_fox_favorite", pd.Series("false", index=current.index))
        updated = current[favorite_values.map(_truthy)].copy()
    else:
        updated = archived.copy()
        for _, row in current.iterrows():
            mask = pd.Series(True, index=updated.index)
            for column in keys:
                mask &= updated.get(column, pd.Series("", index=updated.index)).astype(str).eq(str(row.get(column, "")))
            if _truthy(row.get("red_fox_favorite")):
                updated = updated.loc[~mask].copy()
                updated = pd.concat([updated, row.to_frame().T], ignore_index=True, sort=False)
            elif mask.any():
                # Kickoff providers can revise a start time after qualification.
                # Keep the classification frozen while carrying the latest
                # schedule identity into the eventual handoff.
                for column in ("kickoff_iso", "kickoff_time", "kickoff_sort", "game"):
                    value = str(row.get(column, "")).strip()
                    if value:
                        updated.loc[mask, column] = value
    if updated.empty:
        return
    updated = updated.sort_values("candidate_recorded_at_utc", kind="mergesort").drop_duplicates(keys, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    updated.to_csv(temporary, index=False)
    temporary.replace(path)


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
