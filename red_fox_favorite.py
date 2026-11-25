"""Versioned Red Fox Favorite qualification layered over published Market Reads.

This module intentionally does not classify Market Reads or alter their rank.  It
consumes the two-side evidence already produced by :mod:`anomaly_board` and adds
an independent, binary customer designation plus a prospective audit ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

import pandas as pd


@dataclass(frozen=True)
class FavoriteConfig:
    version: str = "red_fox_favorite_v5"
    ncaaf_version: str = "red_fox_favorite_v8"
    basketball_version: str = "red_fox_favorite_basketball_v2"
    ufc_version: str = "red_fox_favorite_ufc_v1"
    nhl_version: str = "red_fox_favorite_nhl_v1"
    spread_sports: frozenset[str] = frozenset({"nfl", "ncaaf", "cfb", "nba", "ncaab", "cbb"})
    primary_moneyline_sports: frozenset[str] = frozenset({"mlb", "nhl", "ufc"})
    secondary_moneyline_sports: frozenset[str] = frozenset({"nfl", "ncaaf", "cfb", "nba", "ncaab", "cbb"})
    spread_min: float = 1.0
    spread_max: float = 8.0
    nfl_positive_spread_max: float = 10.0
    nfl_extended_spread_max_bets: float = 40.0
    nfl_extended_spread_max_money: float = 49.999
    nfl_extended_spread_min_move: float = 1.0
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
    basketball_spread_move_min: float = 1.5
    basketball_short_favorite_spread_max: float = 1.5
    # An MLB side that opened and remains plus money has not demonstrated the
    # same market conviction as an underdog that was bet through pick'em.  It
    # therefore needs a stronger probability move than the generic market-read
    # threshold before it can become a Red Fox Favorite.
    mlb_still_plus_min_probability_move: float = 5.0
    resistance_min_bets: float = 80.0
    resistance_min_money: float = 60.0
    resistance_max_direction_changes: int = 4
    football_favorite_max_bets: float = 40.0
    football_favorite_max_money: float = 39.999
    football_favorite_max_spread_money: float = 49.999
    football_short_home_max_money: float = 35.0
    football_short_home_max_spread_money: float = 40.0
    football_favorite_max_direction_changes: int = 4
    partial_whipsaw_min_retention: float = 0.65
    partial_whipsaw_max_material_direction_changes: int = 3
    partial_whipsaw_confirmation_minutes: int = 10
    normal_freshness_minutes: int = 30
    whipsaw_freshness_minutes: int = 20
    freeze_disabled_sports: frozenset[str] = frozenset({"ufc", "nhl"})
    visibility_lock_sports: frozenset[str] = frozenset({"nfl", "ncaaf", "cfb", "nba", "ncaab", "cbb", "mlb", "ufc", "nhl"})
    visibility_review_minutes: int = 40
    visibility_lock_minutes: int = 20
    visibility_confirmations: int = 2
    visibility_confirmation_gap_minutes: int = 15
    ncaaf_market_move_min: float = 1.5
    ncaaf_key_number_move_min: float = 1.0
    ncaaf_confirmation_min_gap_minutes: int = 10
    ncaaf_adverse_move_threshold: float = 0.5
    ncaaf_active_market_window_days: int = 7
    ncaaf_partial_whipsaw_min_retention: float = 0.80
    ncaaf_partial_whipsaw_max_material_direction_changes: int = 6
    ncaaf_partial_whipsaw_confirmation_minutes: int = 30
    ncaaf_shadow_split_max: float = 40.0
    ncaaf_shadow_short_split_max: float = 35.0
    ncaaf_sharp_money_max_bets: float = 35.0
    ncaaf_sharp_money_min: float = 46.0
    ncaaf_sharp_money_max: float = 65.0
    ncaaf_sharp_money_min_gap: float = 15.0
    ufc_market_move_min: float = 3.0
    nhl_market_move_min: float = 2.5
    ufc_active_window_hours: float = 36.0
    nhl_active_window_hours: float = 24.0
    primary_moneyline_confirmation_minutes: int = 15
    primary_moneyline_confirmation_max_gap_minutes: int = 30
    primary_moneyline_lock_minutes: int = 30
    primary_moneyline_whipsaw_min_retention: float = 0.80
    primary_moneyline_whipsaw_max_direction_changes: int = 2
    ufc_adverse_probability_points: float = 1.5
    nhl_adverse_probability_points: float = 1.0


CONFIG = FavoriteConfig()
VISIBILITY_INVALIDATED_FAVORITES = frozenset({
    ("ncaaf", "34603681", "SPREAD"),  # Jacksonville State @ Ohio, 2026-09-12
})
RULE_INVALIDATED_FAVORITES = frozenset({
    ("mlb", "34694536", "MONEYLINE"),  # WAS Nationals @ STL Cardinals, 2026-09-20
    ("mlb", "34726372", "MONEYLINE"),  # ATL Braves @ MIA Marlins, 2026-09-26
})
RULE_INVALIDATED_FAVORITE_REASONS = {
    ("mlb", "34694536", "MONEYLINE"): (
        "Favorite removed: MLB side opened and remained plus money without the required "
        "five-point implied-probability move."
    ),
    ("mlb", "34726372", "MONEYLINE"): (
        "Favorite removed after review: Miami opened plus money and crossed to negative "
        "money, a suspended MLB Favorite cohort."
    ),
}
MANUAL_FAVORITE_INCLUSIONS = {
    ("ncaaf", "34708335", "SPREAD"): {
        "team": "San Jose State",
        "reason": (
            "Owner-approved event override: San Jose State is the confirmed low-support "
            "resistance side against concentrated Hawaii pressure."
        ),
    },
}


def _manual_favorite_exclusion(row: pd.Series) -> str:
    """Event-scoped editorial exclusions from Favorite qualification."""
    game = str(row.get("game", "")).lower()
    kickoff = pd.to_datetime(
        row.get("kickoff_iso") or row.get("kickoff_sort"), errors="coerce", utc=True,
    )
    if (
        str(row.get("sport", "")).lower() == "nfl"
        and "seahawks" in game
        and "cardinals" in game
        and pd.notna(kickoff)
        and kickoff.date().isoformat() == "2026-09-20"
    ):
        return "Favorite withheld: Seattle QB Sam Darnold injury affected this game's line movement."
    return ""
FAVORITE_COLUMNS = [
    "red_fox_favorite", "favorite_side", "favorite_pathway", "favorite_rule_version",
    "favorite_first_qualified_at", "favorite_final_qualified_at", "favorite_state",
    "favorite_final_market_read", "favorite_final_market_rank",
    "favorite_supporting_evidence", "favorite_whipsaw_state",
    "favorite_cross_market_state", "favorite_snapshot_id", "favorite_reason",
    "favorite_shadow_freeze_side", "favorite_shadow_freeze_eligible",
    "favorite_shadow_freeze_reason",
    "favorite_originally_qualified", "favorite_late_invalidated",
    "favorite_late_invalidated_at", "favorite_late_invalidated_reason",
]
TRACKING_COLUMNS = [
    "recorded_at", "sport", "game_id", "game", "market_display", "favorite_state",
    "favorite_side", "favorite_pathway", "favorite_rule_version", "bets_pct", "money_pct",
    "first_qualified_at", "final_qualified_at", "open_line", "first_qualified_line", "current_line",
    "final_pregame_line", "final_pregame_market_read", "final_pregame_rank", "board_rank", "reaction", "path",
    "whipsaw_state", "cross_market_state", "supporting_evidence", "snapshot_id",
    "disappearance_reason", "closing_line", "clv", "result",
    "source_observed_at", "entry_observed_at", "transition_reason",
]
REVIEW_COLUMNS = [
    "recorded_at", "sport", "game_id", "game", "market_display", "kickoff_iso",
    "review_state", "review_action", "favorite_side", "current_line", "reason",
    "favorite_rule_version", "source_observed_at",
]
REVIEW_BOARD_COLUMNS = ["favorite_review_state", "favorite_review_reason", "favorite_review_at"]
SHADOW_COLUMNS = [
    "capture_id", "captured_at", "sport", "game_id", "game", "kickoff_iso",
    "minutes_before_start", "market_display", "candidate_side", "candidate_decision",
    "candidate_reason", "favorite_pathway", "favorite_rule_version", "open_line",
    "current_line", "bets_pct", "money_pct", "reaction", "path", "data_badge",
    "observation_count", "line_direction_changes", "return_toward_open",
    "active_worsening_reversal", "whipsaw_state", "cross_market_state", "snapshot_id",
    "whipsaw_retention_ratio", "material_direction_changes",
    "whipsaw_confirmation_ready", "whipsaw_confirmation_minutes",
    "shadow_cohort", "shadow_eligible",
    "legacy_candidate_decision", "new_candidate_decision",
    "ncaaf_move_gate_passed", "ncaaf_market_move_status",
    "open_observed_at", "open_age_days", "stale_open_status",
    "qualifying_line", "t20_line", "closing_spread",
    "legacy_split_45_passed", "shadow_split_40_passed",
    "shadow_short_split_35_passed", "strict_whipsaw_shadow_passed",
    "game_result", "grade", "units_won_lost", "clv",
    "decision_reason_detail",
]


NOV24_SPORTS = frozenset({"ncaab", "nba", "nhl"})
NBA_PUBLIC_PATH = "nba_short_public_neutral_ml"
NBA_PUBLIC_VERSION = "red_fox_favorite_nba_public_neutral_v1"
NBA_PUBLIC_FLAG = "RED_FOX_NBA_PUBLIC_NEUTRAL"
NBA_LEGACY_HOLD_FLAG = "RED_FOX_NBA_HOLD_LEGACY"
NOV24_VERSION = "red_fox_favorite_20261124"
CFB_PUBLIC_FOLLOW = "cfb_public_spread_follow"
CFB_HIGH_MONEY = "cfb_low_ticket_high_money_spread"
CFB_SPREAD_VERSIONS = {
    CFB_PUBLIC_FOLLOW: "red_fox_favorite_cfb_public_follow_v1",
    CFB_HIGH_MONEY: "red_fox_favorite_cfb_high_money_v1",
}
CFB_SPREAD_FLAGS = {
    CFB_PUBLIC_FOLLOW: "RED_FOX_CFB_PUBLIC_FOLLOW",
    CFB_HIGH_MONEY: "RED_FOX_CFB_HIGH_MONEY",
}


def cfb_spread_favorite(row: pd.Series) -> bool:
    return (
        str(row.get("sport", "")).lower() in {"ncaaf", "cfb"}
        and str(row.get("market_display", "")).upper() == "SPREAD"
        and CFB_SPREAD_VERSIONS.get(row.get("favorite_pathway")) == row.get("favorite_rule_version")
        and row.get("favorite_pathway") in CFB_SPREAD_VERSIONS
    )


def nba_public_favorite(row: pd.Series | None) -> bool:
    return (row is not None and row.get("sport") == "nba" and row.get("market_display") == "MONEYLINE"
            and row.get("favorite_rule_version") == NBA_PUBLIC_VERSION and row.get("favorite_pathway") == NBA_PUBLIC_PATH)


def nba_release_config() -> tuple[bool, bool]:
    return _truthy(os.getenv(NBA_PUBLIC_FLAG)), _truthy(os.getenv(NBA_LEGACY_HOLD_FLAG))


def apply_red_fox_favorites(board: pd.DataFrame, as_of=None, *, nov24_goalies: dict | None = None) -> pd.DataFrame:
    """Add deterministic Favorite fields without changing row order or rank."""
    if board is None:
        return pd.DataFrame(columns=FAVORITE_COLUMNS)
    result = board.copy()
    for column in FAVORITE_COLUMNS:
        result[column] = ""
    result["red_fox_favorite"] = "false"
    result["favorite_rule_version"] = result.get(
        "sport", pd.Series("", index=result.index)
    ).map(_favorite_rule_version)
    result["favorite_state"] = "not_qualified"
    result["_favorite_legacy_qualified"] = "false"
    result["_favorite_legacy_side"] = ""
    result["_favorite_legacy_reason"] = ""
    result["_ncaaf_move_gate_passed"] = ""
    result["_ncaaf_market_move_status"] = ""
    result["_ncaaf_decision_reason"] = ""
    if result.empty:
        return result

    captured_at = _timestamp(as_of)
    groups = {
        (str(sport).lower(), str(game_id)): group
        for (sport, game_id), group in result.groupby(["sport", "game_id"], dropna=False, sort=False)
    }
    opener_blocked_games = set()
    for index, row in result.iterrows():
        identity = (
            str(row.get("sport", "")).lower(),
            str(row.get("game_id", "")),
            str(row.get("market_display", "")).upper(),
        )
        shadow_freeze = _shadow_freeze_candidate(
            row, groups.get((str(row.get("sport", "")).lower(), str(row.get("game_id", ""))))
        )
        if shadow_freeze:
            result.at[index, "favorite_shadow_freeze_side"] = shadow_freeze["side"]
            result.at[index, "favorite_shadow_freeze_eligible"] = "true"
            result.at[index, "favorite_shadow_freeze_reason"] = shadow_freeze["reason"]
        if identity in VISIBILITY_INVALIDATED_FAVORITES:
            result.at[index, "favorite_reason"] = (
                "Favorite qualification withheld: audited final-hour visibility failure."
            )
            continue
        manual_exclusion = _manual_favorite_exclusion(row)
        if manual_exclusion:
            result.at[index, "favorite_reason"] = manual_exclusion
            continue
        game_rows = groups.get((str(row.get("sport", "")).lower(), str(row.get("game_id", ""))))
        legacy = _qualify_market(row, game_rows, enforce_ncaaf_policy=False, at=captured_at)
        if legacy:
            result.at[index, "_favorite_legacy_qualified"] = "true"
            result.at[index, "_favorite_legacy_side"] = legacy["side"]
            result.at[index, "_favorite_legacy_reason"] = legacy["reason"]
        elif str(row.get("sport", "")).lower() in {"ncaaf", "cfb"}:
            supported = next(
                (item for item in _sides(row) if _side_identity(item.get("flagged_side")) == _side_identity(row.get("supported_side"))),
                None,
            )
            rejection = _legacy_rejection_reason(row, supported, game_rows)
            result.at[index, "_favorite_legacy_reason"] = rejection
            result.at[index, "favorite_reason"] = rejection
        failures = {}
        decision = _qualify_market(row, game_rows, at=captured_at, failures=failures)
        if not decision and nov24_goalies is not None and identity[0] in NOV24_SPORTS:
            decision = _nov24_decision(row, game_rows, captured_at, nov24_goalies)
        if not decision:
            supported_identity = _side_identity(row.get("supported_side", ""))
            side_failure = failures.get(supported_identity) or failures.get("")
            if side_failure:
                result.at[index, "_favorite_gate_reason"] = side_failure
            if legacy and str(row.get("sport", "")).lower() in {"ncaaf", "cfb"}:
                side = next(
                    (item for item in _sides(row) if _side_identity(item.get("flagged_side")) == _side_identity(legacy["side"])),
                    None,
                )
                if side is not None:
                    gate, cross_state = _ncaaf_gate_for_market(row, side, game_rows)
                    result.at[index, "_ncaaf_move_gate_passed"] = str(gate["move_gate_passed"]).lower()
                    result.at[index, "_ncaaf_market_move_status"] = gate["market_move_status"]
                    result.at[index, "_ncaaf_decision_reason"] = gate["reason"]
                    result.at[index, "favorite_cross_market_state"] = cross_state
                    result.at[index, "favorite_reason"] = gate["reason"]
            elif str(row.get("sport", "")).lower() in {"nba", "ncaab", "cbb"}:
                supported = next(
                    (
                        item for item in _sides(row)
                        if _side_identity(item.get("flagged_side"))
                        == _side_identity(row.get("supported_side"))
                    ),
                    None,
                )
                if supported is None:
                    result.at[index, "favorite_reason"] = (
                        "Basketball Favorite withheld: no exact currently confirmed Supported Side."
                    )
                else:
                    gate, cross_state = _basketball_gate_for_market(row, supported, game_rows)
                    result.at[index, "favorite_cross_market_state"] = cross_state
                    result.at[index, "favorite_reason"] = gate["reason"]
            elif str(row.get("sport", "")).lower() in {"ufc", "nhl"}:
                result.at[index, "favorite_reason"] = _primary_moneyline_rejection_reason(
                    row, game_rows, captured_at,
                )
            continue
        gate = decision.get("ncaaf_gate", {})
        if gate:
            result.at[index, "_ncaaf_move_gate_passed"] = str(gate["move_gate_passed"]).lower()
            result.at[index, "_ncaaf_market_move_status"] = gate["market_move_status"]
            result.at[index, "_ncaaf_decision_reason"] = gate["reason"]
        current_mismatch = _truthy(row.get("cross_market_mismatch"))
        opener_mismatch = _truthy(row.get("cross_market_opener_mismatch_verified"))
        if current_mismatch or opener_mismatch:
            result.at[index, "favorite_cross_market_state"] = "mismatch"
            result.at[index, "favorite_reason"] = "Confirmed Cross-Market Mismatch withheld Favorite qualification."
            result.at[index, "_favorite_gate_reason"] = result.at[index, "favorite_reason"]
            if opener_mismatch and not current_mismatch:
                opener_blocked_games.add((str(row.get("sport", "")).lower(), str(row.get("game_id", ""))))
            continue
        if decision.get("nov24") and not str(row.get("supported_side") or "").strip():
            # The addition provides direct directional evidence. An existing
            # contradictory Supported Side is never overwritten by this path.
            result.at[index, "supported_side"] = decision["side"]
            result.at[index, "_nov24_added_support"] = True
        snapshot_id = _snapshot_id(row, captured_at)
        result.at[index, "red_fox_favorite"] = "true"
        result.at[index, "favorite_side"] = decision["side"]
        result.at[index, "favorite_pathway"] = decision["pathway"]
        result.at[index, "favorite_rule_version"] = (
            NOV24_VERSION if decision.get("nov24") else _favorite_rule_version(row.get("sport", ""))
        )
        result.at[index, "favorite_first_qualified_at"] = captured_at
        result.at[index, "favorite_final_qualified_at"] = captured_at
        result.at[index, "favorite_state"] = "qualified"
        result.at[index, "favorite_final_market_read"] = decision["market_read"]
        result.at[index, "favorite_final_market_rank"] = str(row.get("board_rank", ""))
        result.at[index, "favorite_supporting_evidence"] = json.dumps(decision["evidence"], separators=(",", ":"))
        result.at[index, "favorite_whipsaw_state"] = decision["whipsaw_state"]
        result.at[index, "favorite_cross_market_state"] = decision["cross_market_state"]
        result.at[index, "favorite_snapshot_id"] = snapshot_id
        result.at[index, "favorite_reason"] = decision["reason"]
        result.at[index, "favorite_originally_qualified"] = "true"
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
    result = _prefer_confirmed_football_moneyline(result)
    result = _prefer_basketball_moneyline_expression(result)
    if nov24_goalies is not None:
        picked = result["red_fox_favorite"].map(_truthy)
        added = result["favorite_rule_version"].eq(NOV24_VERSION)
        existing_games = set(zip(result.loc[picked & ~added, "sport"], result.loc[picked & ~added, "game_id"]))
        for index, row in result.loc[picked & added].iterrows():
            if (row["sport"], row["game_id"]) in existing_games:
                _clear_favorite(result, index, "Existing Favorite takes precedence over the new pathway.")
    return result


def update_favorite_tracking(
    board: pd.DataFrame, data_dir: Path, as_of=None, *,
    market_history: pd.DataFrame | None = None,
    cfb_enabled: frozenset[str] | None = None,
    nba_config: tuple[bool, bool] | None = None,
) -> pd.DataFrame:
    """Persist every prospective snapshot and carry first qualification forward."""
    current = board.copy()
    at = _timestamp(as_of)
    path = Path(data_dir) / "red_fox_favorite_tracking.csv"
    history = _read_tracking(path)
    if cfb_enabled is None:
        cfb_enabled = frozenset(path for path, flag in CFB_SPREAD_FLAGS.items() if _truthy(os.getenv(flag)))
    current = _apply_cfb_spread_pathways(current, history, market_history, at, cfb_enabled)
    nba_config = nba_release_config() if nba_config is None else nba_config
    current = _apply_nba_release(current, history, at, *nba_config)
    current = _apply_visibility_lock(current, history, Path(data_dir), at, nba_hold_legacy=nba_config[1])
    current = _apply_manual_favorite_inclusions(current, at)
    _resolve_cfb_spread_conflicts(current, at)
    keys = ["sport", "game_id", "market_display"]
    started_frozen_keys: set[tuple[str, str, str]] = set()
    archive_path = Path(data_dir) / "red_fox_favorite_freeze_candidates.csv"
    try:
        archive = pd.read_csv(archive_path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError):
        archive = pd.DataFrame()
    if not archive.empty and all(column in archive for column in (*keys, "kickoff_iso")):
        now = pd.Timestamp(at)
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        kickoff = pd.to_datetime(archive["kickoff_iso"], errors="coerce", utc=True)
        official = archive.get("red_fox_favorite", pd.Series("false", index=archive.index)).map(_truthy)
        for _, frozen in archive.loc[official & kickoff.notna() & kickoff.le(now)].iterrows():
            started_frozen_keys.add(tuple(str(frozen.get(column, "")) for column in keys))
    previous = {}
    first_entries = {}
    if not history.empty:
        ordered = history.sort_values("recorded_at", kind="mergesort")
        for _, prior in ordered.iterrows():
            key = tuple(str(prior.get(column, "")) for column in keys)
            previous[key] = prior
            if (key[2] == "SPREAD" and prior.get("favorite_state") == "qualified"
                    and prior.get("recorded_at") == prior.get("first_qualified_at")):
                identity = _favorite_review_identity(prior)
                if all(identity):
                    first_entries.setdefault((*key, *identity), prior)

    records = []
    current_keys = set()
    for index, row in current.iterrows():
        key = tuple(str(row.get(column, "")) for column in keys)
        current_keys.add(key)
        prior = previous.get(key)
        favorite = _truthy(row.get("red_fox_favorite"))
        same_identity = prior is not None and _favorite_review_identity(row) == _favorite_review_identity(prior)
        if favorite and same_identity:
            prior_first = str(prior.get("first_qualified_line", "")).strip()
            prior_at = str(prior.get("recorded_at", "")).strip()
            first_at = str(prior.get("first_qualified_at", "")).strip() or prior_at or at
            current.at[index, "favorite_first_qualified_at"] = first_at
            first_capture = first_entries.get((*key, *_favorite_review_identity(row)))
            verified, entry_time = tracked_entry_quote(first_capture) if first_capture is not None else ("", "")
            first_line = verified or prior_first or favorite_side_quote(row)
        else:
            first_line = favorite_side_quote(row) if favorite else ""
        same_snapshot = (
            favorite and prior is not None
            and str(prior.get("favorite_state", "")) == "qualified"
            and str(prior.get("snapshot_id", "")) == str(row.get("favorite_snapshot_id", ""))
        )
        if favorite:
            if not same_snapshot or not same_identity:
                record = _tracking_record(current.loc[index], at, first_line)
                record["entry_observed_at"] = (
                    entry_time or str(prior.get("entry_observed_at", "")) if same_identity else record["source_observed_at"]
                )
                records.append(record)
        elif prior is not None and str(prior.get("favorite_state", "")) == "qualified":
            record = {column: str(prior.get(column, "")) for column in TRACKING_COLUMNS}
            sport = str(row.get("sport", "")).lower()
            rejection = str(row.get("favorite_reason", "")).strip() if sport in {"ufc", "nhl"} else ""
            if not any(token in rejection.lower() for token in ("withheld", "failed", "removal", "unavailable")):
                rejection = ""
            review_reason = str(row.get("favorite_review_reason", "")).strip()
            gate_reason = str(row.get("_favorite_gate_reason", "")).strip()
            if gate_reason and gate_reason.lower() != "nan":
                review_reason = " | ".join(dict.fromkeys(filter(None, (review_reason, gate_reason))))
            record.update(
                recorded_at=at,
                favorite_state="not_qualified",
                disappearance_reason=rejection or "current qualification gates no longer satisfied",
                source_observed_at=_review_source_observed_at(row, prior.get("favorite_side", "")),
                transition_reason=review_reason or str(row.get("favorite_reason", "")).strip(),
            )
            records.append(record)

    # A previously qualifying market can leave the board because it started or
    # ceased to meet a gate. Record that transition without rewriting its frozen
    # Live & Recent row.
    for key, prior in previous.items():
        if key in current_keys or str(prior.get("favorite_state", "")) != "qualified":
            continue
        # Kickoff removes a market from the live pregame board, but it must not
        # rewrite the official qualification history as a Favorite fall-off.
        if key in started_frozen_keys:
            continue
        record = {column: str(prior.get(column, "")) for column in TRACKING_COLUMNS}
        record.update(recorded_at=at, favorite_state="not_qualified",
                      disappearance_reason="no longer published or eligible",
                      source_observed_at="", transition_reason="Market absent from the evaluated board.")
        records.append(record)

    if records:
        new_records = pd.DataFrame(records)
        updated = new_records if history.empty else pd.concat([history, new_records], ignore_index=True, sort=False)
        updated = updated.reindex(columns=TRACKING_COLUMNS)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + ".tmp")
        updated.to_csv(temporary, index=False)
        temporary.replace(path)
    _update_candidate_shadow(current, Path(data_dir), at)
    _update_freeze_candidates(current, Path(data_dir), at)
    return current.drop(columns=[
        "_visibility_restored", "_favorite_legacy_qualified", "_favorite_legacy_side",
        "_favorite_legacy_reason", "_ncaaf_move_gate_passed",
        "_ncaaf_market_move_status", "_ncaaf_decision_reason", "_favorite_gate_reason",
        "_nba_release_mismatch",
    ], errors="ignore")


def _apply_manual_favorite_inclusions(frame: pd.DataFrame, at: str) -> pd.DataFrame:
    """Apply narrowly scoped owner decisions after the generic closing lock."""
    result = frame.copy()
    for index, row in result.iterrows():
        key = (
            str(row.get("sport", "")).lower(),
            str(row.get("game_id", "")),
            str(row.get("market_display", "")).upper(),
        )
        override = MANUAL_FAVORITE_INCLUSIONS.get(key)
        if not override:
            continue
        supported = str(row.get("supported_side", "")).strip()
        if _side_identity(supported) != _side_identity(override["team"]):
            continue
        side = next(
            (item for item in _sides(row) if _side_identity(item.get("flagged_side")) == _side_identity(override["team"])),
            None,
        )
        if side is None:
            continue
        evidence = {
            "reaction": str(side.get("reaction", "")),
            "path": str(side.get("path", "")),
            "bets_pct": _number(side.get("bets_pct")),
            "money_pct": _number(side.get("money_pct")),
            "open_line": str(side.get("open_line", "")),
            "current_line": str(side.get("current_line", "")),
            "observation_count": int(_number(side.get("observation_count")) or 0),
            "cross_market": "confirmation",
            "manual_event_override": True,
        }
        result.at[index, "red_fox_favorite"] = "true"
        result.at[index, "favorite_side"] = str(side.get("flagged_side", supported))
        result.at[index, "favorite_pathway"] = "manual_event_override"
        result.at[index, "favorite_rule_version"] = _favorite_rule_version(row.get("sport", ""))
        result.at[index, "favorite_first_qualified_at"] = at
        result.at[index, "favorite_final_qualified_at"] = at
        result.at[index, "favorite_state"] = "qualified"
        result.at[index, "favorite_final_market_read"] = str(side.get("anomaly_chips") or side.get("reaction", ""))
        result.at[index, "favorite_final_market_rank"] = str(row.get("board_rank", ""))
        result.at[index, "favorite_supporting_evidence"] = json.dumps(evidence, separators=(",", ":"))
        result.at[index, "favorite_whipsaw_state"] = "manual_event_override"
        result.at[index, "favorite_cross_market_state"] = "confirmation"
        result.at[index, "favorite_snapshot_id"] = _snapshot_id(row, at)
        result.at[index, "favorite_reason"] = override["reason"]
        result.at[index, "favorite_originally_qualified"] = "true"
        result.at[index, "favorite_review_state"] = "applied_override"
        result.at[index, "favorite_review_reason"] = override["reason"]
        result.at[index, "favorite_review_at"] = at
    return result


def _update_candidate_shadow(current: pd.DataFrame, data_dir: Path, at: str) -> None:
    """Append every published Supported Side evaluation, accepted or rejected."""
    path = data_dir / "red_fox_favorite_shadow.csv"
    try:
        history = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() and path.stat().st_size else pd.DataFrame(columns=SHADOW_COLUMNS)
    except (OSError, pd.errors.EmptyDataError):
        history = pd.DataFrame(columns=SHADOW_COLUMNS)
    records = []
    captured = pd.Timestamp(at)
    captured = captured.tz_localize("UTC") if captured.tzinfo is None else captured.tz_convert("UTC")
    for _, row in current.iterrows():
        supported = str(row.get("supported_side", "")).strip()
        shadow_side = str(row.get("favorite_shadow_freeze_side", "")).strip()
        candidate_specs = []
        if supported:
            candidate_specs.append((supported, "favorite_rules", ""))
        if shadow_side and _side_identity(shadow_side) != _side_identity(supported):
            candidate_specs.append((shadow_side, "nfl_freeze_75_79", row.get("favorite_shadow_freeze_reason", "")))
        if not candidate_specs:
            continue
        kickoff = pd.to_datetime(row.get("kickoff_iso", ""), errors="coerce", utc=True)
        minutes = "" if pd.isna(kickoff) else f"{round((kickoff - captured).total_seconds() / 60.0, 3):g}"
        for candidate_side, cohort, shadow_reason in candidate_specs:
            candidate = next((side for side in _sides(row) if _side_identity(side.get("flagged_side")) == _side_identity(candidate_side)), None)
            if candidate is None:
                continue
            accepted = cohort == "favorite_rules" and _truthy(row.get("red_fox_favorite")) and _side_identity(row.get("favorite_side")) == _side_identity(candidate_side)
            capture_key = "|".join([
                at, str(row.get("sport", "")), str(row.get("game_id", "")),
                str(row.get("market_display", "")), candidate_side, cohort,
            ])
            context = _parts(candidate.get("context_chips"))
            whipsaw = "recovered" if _truthy(candidate.get("whipsaw_recovered")) or "Whipsaw Recovered" in context else (
                "active_or_partial" if str(candidate.get("path", "")) == "Whipsaw" else "none"
            )
            sport = str(row.get("sport", "")).lower()
            bets = _number(candidate.get("bets_pct"))
            money = _number(candidate.get("money_pct"))
            current_line = _line_value(candidate.get("current_line"), str(row.get("market_display", "")).upper())
            open_at = pd.to_datetime(candidate.get("open_observed_at", ""), errors="coerce", utc=True)
            open_age = ""
            stale_open = "unknown"
            if pd.notna(kickoff) and pd.notna(open_at):
                age_days = (kickoff - open_at).total_seconds() / 86400.0
                open_age = f"{round(age_days, 3):g}"
                stale_open = "true" if age_days > CONFIG.ncaaf_active_market_window_days else "false"
            legacy_accepted = (
                cohort == "favorite_rules"
                and _truthy(row.get("_favorite_legacy_qualified"))
                and _side_identity(row.get("_favorite_legacy_side")) == _side_identity(candidate_side)
            )
            strict_whipsaw = (
                whipsaw == "none"
                or (
                    (_number(candidate.get("whipsaw_retention_ratio")) or 0) >= 0.90
                    and (_number(candidate.get("material_direction_changes")) or 0) <= 1
                )
            )
            records.append({
            "capture_id": hashlib.sha256(capture_key.encode("utf-8")).hexdigest()[:24],
            "captured_at": at, "sport": row.get("sport", ""), "game_id": row.get("game_id", ""),
            "game": row.get("game", ""), "kickoff_iso": row.get("kickoff_iso", ""),
            "minutes_before_start": minutes, "market_display": row.get("market_display", ""),
            "candidate_side": candidate_side,
            "candidate_decision": "shadow_eligible" if cohort != "favorite_rules" else ("accepted" if accepted else "rejected"),
            "candidate_reason": shadow_reason or row.get("favorite_reason", "") or (
                "Accepted by current Favorite rules." if accepted else "Rejected by current Favorite rules."
            ),
            "favorite_pathway": row.get("favorite_pathway", ""),
            "favorite_rule_version": row.get(
                "favorite_rule_version", _favorite_rule_version(row.get("sport", ""))
            ),
            "open_line": candidate.get("open_line", ""), "current_line": candidate.get("current_line", ""),
            "bets_pct": candidate.get("bets_pct", ""), "money_pct": candidate.get("money_pct", ""),
            "reaction": candidate.get("reaction", ""), "path": candidate.get("path", ""),
            "data_badge": candidate.get("data_badge", ""),
            "observation_count": candidate.get("observation_count", ""),
            "line_direction_changes": candidate.get("line_dir_changes", ""),
            "return_toward_open": candidate.get("return_toward_open", ""),
            "active_worsening_reversal": candidate.get("active_worsening_reversal", ""),
            "whipsaw_state": whipsaw,
            "cross_market_state": row.get("favorite_cross_market_state", ""),
            "snapshot_id": row.get("favorite_snapshot_id", ""),
            "whipsaw_retention_ratio": candidate.get("whipsaw_retention_ratio", ""),
            "material_direction_changes": candidate.get("material_direction_changes", ""),
            "whipsaw_confirmation_ready": candidate.get("whipsaw_confirmation_ready", ""),
            "whipsaw_confirmation_minutes": candidate.get("whipsaw_confirmation_minutes", ""),
            "shadow_cohort": cohort if cohort != "favorite_rules" else "",
            "shadow_eligible": "true" if cohort != "favorite_rules" else "false",
            "legacy_candidate_decision": "accepted" if legacy_accepted else "rejected",
            "new_candidate_decision": "accepted" if accepted else "rejected",
            "ncaaf_move_gate_passed": row.get("_ncaaf_move_gate_passed", "") if sport in {"ncaaf", "cfb"} else "",
            "ncaaf_market_move_status": row.get("_ncaaf_market_move_status", "") if sport in {"ncaaf", "cfb"} else "",
            "open_observed_at": candidate.get("open_observed_at", ""),
            "open_age_days": open_age,
            "stale_open_status": stale_open,
            "qualifying_line": candidate.get("current_line", "") if legacy_accepted or accepted else "",
            "t20_line": candidate.get("current_line", "") if minutes and 0 <= float(minutes) <= CONFIG.visibility_lock_minutes else "",
            "closing_spread": "",
            "legacy_split_45_passed": str(bool(bets is not None and money is not None and bets <= 45 and money <= 45)).lower(),
            "shadow_split_40_passed": str(bool(
                bets is not None and money is not None
                and bets <= CONFIG.ncaaf_shadow_split_max and money <= CONFIG.ncaaf_shadow_split_max
            )).lower(),
            "shadow_short_split_35_passed": str(bool(
                bets is not None and money is not None and current_line is not None
                and abs(current_line) <= 3
                and bets <= CONFIG.ncaaf_shadow_short_split_max
                and money <= CONFIG.ncaaf_shadow_short_split_max
            )).lower(),
            "strict_whipsaw_shadow_passed": str(strict_whipsaw).lower(),
            "game_result": "", "grade": "", "units_won_lost": "", "clv": "",
            "decision_reason_detail": shadow_reason or row.get("_ncaaf_decision_reason", "") or row.get("favorite_reason", ""),
        })
    if not records:
        return
    new = pd.DataFrame(records, columns=SHADOW_COLUMNS)
    updated = new if history.empty else pd.concat(
        [history.reindex(columns=SHADOW_COLUMNS), new], ignore_index=True,
    )
    updated = updated.drop_duplicates("capture_id", keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    updated.to_csv(temporary, index=False)
    temporary.replace(path)


def _visibility_review_minutes(sport: str) -> float:
    return 40.0 if sport in {"ufc", "nhl"} else float(CONFIG.visibility_review_minutes)


def _visibility_lock_minutes(sport: str) -> float:
    if sport in {"ufc", "nhl"}:
        return float(CONFIG.primary_moneyline_lock_minutes)
    return float(CONFIG.visibility_lock_minutes)


def _apply_visibility_lock(
    current: pd.DataFrame,
    history: pd.DataFrame,
    data_dir: Path,
    at: str,
    *, nba_hold_legacy: bool = False,
) -> pd.DataFrame:
    """Apply the two-stage T-40 review window and T-20 display lock with safety suppression."""
    if current is None or "kickoff_iso" not in current:
        return current
    result = current.copy()
    for column in REVIEW_BOARD_COLUMNS:
        result[column] = ""
    now = pd.Timestamp(at)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    archived_path = data_dir / "red_fox_favorite_freeze_candidates.csv"
    try:
        archived = (
            pd.read_csv(archived_path, dtype=str, keep_default_na=False)
            if archived_path.exists() and archived_path.stat().st_size
            else pd.DataFrame()
        )
    except (OSError, pd.errors.EmptyDataError):
        archived = pd.DataFrame()
    review_path = data_dir / "red_fox_favorite_review.csv"
    reviews = _read_review(review_path)
    review_records: list[dict] = []

    keys = ("sport", "game_id", "market_display")
    extra_slates = {}
    if not archived.empty and "favorite_rule_version" in archived:
        for _, row in archived.loc[archived["favorite_rule_version"].eq(NOV24_VERSION)].iterrows():
            start = pd.to_datetime(row.get("kickoff_iso"), errors="coerce", utc=True)
            if pd.notna(start):
                slate = (row.get("sport"), start.tz_convert("America/New_York").date())
                extra_slates[slate] = str(row.get("game_id"))
    # A fully qualified cross-market preference is an atomic change in wager
    # expression, not threshold flicker.  Preserve the ordinary freshness and
    # safety checks, but do not let the per-market visibility lock temporarily
    # restore the spread or suppress the replacement moneyline.
    translation_roles = {}
    for _, row in result.iterrows():
        key = tuple(str(row.get(column, "")) for column in keys)
        reason = str(row.get("favorite_reason", ""))
        if reason.startswith("Selected over the matching "):
            translation_roles[key] = "preferred_moneyline"
        elif reason.startswith("Favorite expressed on the confirmed moneyline"):
            translation_roles[key] = "superseded_spread"
    current_keys = {
        tuple(str(row.get(column, "")) for column in keys)
        for _, row in result.iterrows()
    }
    if not archived.empty and all(column in archived for column in (*keys, "kickoff_iso")):
        additions = []
        for _, row in archived.iterrows():
            key = tuple(str(row.get(column, "")) for column in keys)
            if key in current_keys:
                continue
            identity = (key[0].lower(), key[1], key[2].upper())
            if (identity in VISIBILITY_INVALIDATED_FAVORITES
                    or _manual_favorite_exclusion(row)
                    or identity[0] not in CONFIG.visibility_lock_sports):
                continue
            kickoff = pd.to_datetime(row.get("kickoff_iso", ""), errors="coerce", utc=True)
            if pd.isna(kickoff):
                continue
            minutes = (kickoff - now).total_seconds() / 60
            active = _official_active(history, key, now)
            if not active or minutes < 0:
                continue
            stale_reason = _freshness_failure(None, row, now)
            if stale_reason:
                if not _review_already_recorded(reviews, key, "late_invalidated"):
                    review_records.append(_review_record(row, at, "late_invalidated", "removal", stale_reason))
                continue
            review_minutes = _visibility_review_minutes(identity[0])
            lock_minutes = _visibility_lock_minutes(identity[0])
            if minutes > review_minutes:
                reason = "Favorite market temporarily absent; awaiting a second consecutive nonqualifying scrape."
                if _review_confirmed(reviews, key, "removal", now, row):
                    review_records.append(_review_record(row, at, "applied", "removal", _latest_review_reason(reviews, key) or reason))
                    continue
                held = row.copy()
                held["favorite_reason"] = "Favorite held through a single missing-market capture."
                held["favorite_review_state"] = "pending_removal"
                held["favorite_review_reason"] = reason
                held["favorite_review_at"] = at
                held["_visibility_restored"] = "true"
                additions.append(held)
                review_records.append(_review_record(row, at, "pending", "removal", reason))
            elif minutes <= review_minutes:
                if (identity[0] in {"ufc", "nhl"} or cfb_spread_favorite(row)) and minutes <= lock_minutes:
                    reason = f"Mandatory T−{lock_minutes:g} {identity[0].upper()} requalification failed: market unavailable."
                    if not _review_already_recorded(reviews, key, "late_invalidated"):
                        review_records.append(_review_record(row, at, "late_invalidated", "removal", reason))
                    if cfb_spread_favorite(row):
                        invalidated = row.copy()
                        invalidated["red_fox_favorite"] = "false"
                        invalidated["favorite_state"] = "late_invalidated"
                        invalidated["favorite_late_invalidated"] = "true"
                        invalidated["favorite_late_invalidated_at"] = at
                        invalidated["favorite_late_invalidated_reason"] = reason
                        invalidated["favorite_reason"] = reason
                        additions.append(invalidated)
                    continue
                held = row.copy()
                held["favorite_reason"] = "Favorite retained through the closing review window."
                held["_visibility_restored"] = "true"
                additions.append(held)
        if additions:
            result = pd.concat([result, pd.DataFrame(additions)], ignore_index=True, sort=False)

    for index, row in result.iterrows():
        kickoff = pd.to_datetime(row.get("kickoff_iso", ""), errors="coerce", utc=True)
        if pd.isna(kickoff):
            continue
        minutes = (kickoff - now).total_seconds() / 60
        key = tuple(str(row.get(column, "")) for column in keys)
        identity = (key[0].lower(), key[1], key[2].upper())
        if identity in VISIBILITY_INVALIDATED_FAVORITES:
            _clear_favorite(result, index, "Favorite qualification withheld: audited final-hour visibility failure.")
            continue
        manual_exclusion = _manual_favorite_exclusion(row)
        if manual_exclusion:
            _clear_favorite(result, index, manual_exclusion)
            continue
        if identity[0] not in CONFIG.visibility_lock_sports:
            continue
        review_minutes = _visibility_review_minutes(identity[0])
        lock_minutes = _visibility_lock_minutes(identity[0])
        raw_favorite = _truthy(row.get("red_fox_favorite"))
        active = _official_active(history, key, now)
        source = _archived_row(archived, key)
        if minutes < 0 and active:
            _restore_favorite(result, index, source)
            result.at[index, "favorite_reason"] = "Official Favorite retained after kickoff handoff."
            _set_review(result, index, "locked_at_start", result.at[index, "favorite_reason"], at)
            continue
        stale_reason = _freshness_failure(row, source, now)
        if stale_reason and (raw_favorite or active):
            _late_invalidate_favorite(result, index, stale_reason, at)
            _set_review(result, index, "late_invalidated", stale_reason, at)
            if not _review_already_recorded(reviews, key, "late_invalidated"):
                review_records.append(_review_record(row, at, "late_invalidated", "removal", stale_reason))
            continue

        if (raw_favorite and active and source is not None and cfb_spread_favorite(source)
                and (_side_identity(row.get("favorite_side")) != _side_identity(source.get("favorite_side"))
                     or (cfb_spread_favorite(row) and _favorite_review_identity(row) != _favorite_review_identity(source)))):
            _clear_favorite(result, index, "Awaiting qualification of a different CFB pathway or side.")
            row = result.loc[index].copy()
            raw_favorite = False

        # A legacy qualification of the same side keeps its own version,
        # including when it supersedes an experimental spread inside review.
        if (raw_favorite and active and source is not None and cfb_spread_favorite(source)
                and not cfb_spread_favorite(row) and 0 <= minutes <= review_minutes):
            hard, adverse, reason = _material_removal(row, source)
            if hard or (minutes <= lock_minutes and adverse):
                _late_invalidate_favorite(result, index, reason, at)
                _set_review(result, index, "late_invalidated", reason, at)
                review_records.append(_review_record(row, at, "late_invalidated", "removal", reason, identity_row=source))
            else:
                _clear_pending_review(reviews, review_records, row, key, at)
            continue

        if key in translation_roles:
            _clear_pending_review(reviews, review_records, row, key, at)
            continue

        if (raw_favorite and active and source is not None
                and str(source.get("favorite_rule_version")) == NOV24_VERSION
                and _side_identity(row.get("favorite_side")) == _side_identity(source.get("favorite_side"))):
            result.at[index, "favorite_rule_version"] = NOV24_VERSION
            if identity[0] == "nhl":
                old_pair = json.loads(str(source.get("favorite_supporting_evidence", "{}"))).get("confirmed_goalies")
                new_pair = json.loads(str(row.get("favorite_supporting_evidence", "{}"))).get("confirmed_goalies")
                if old_pair and new_pair and old_pair != new_pair:
                    reason = "11/24 NHL Favorite invalidated after a confirmed starting-goalie change."
                    _late_invalidate_favorite(result, index, reason, at)
                    _set_review(result, index, "late_invalidated", reason, at)
                    review_records.append(_review_record(row, at, "late_invalidated", "removal", reason))
                    continue

        if raw_favorite and not active and str(row.get("favorite_rule_version")) == NOV24_VERSION:
            # Extra paths can qualify before closing review only. Keep the
            # old closing-window and kickoff safeguards for all official rows.
            slate = (identity[0], kickoff.tz_convert("America/New_York").date())
            if minutes <= review_minutes or extra_slates.get(slate, identity[1]) != identity[1]:
                _clear_favorite(result, index, "11/24 pathway held by the closing window or slate limit.")
                _clear_pending_review(reviews, review_records, row, key, at)
                continue
            evidence = json.loads(str(row["favorite_supporting_evidence"]))
            signature = [_side_identity(row["favorite_side"]), evidence.get("confirmed_goalies")]
            stamps = evidence["confirmation_sources"]
            reason = "11/24 confirmation:" + json.dumps([signature, stamps], separators=(",", ":"))
            pending = _review_rows(reviews, key)
            if not pending.empty:
                last = pending.iloc[-1]
                previous = pd.to_datetime(last.get("recorded_at"), errors="coerce", utc=True)
                gap = (now - previous).total_seconds() / 60 if pd.notna(previous) else -1
                last_reason = str(last.get("reason", ""))
                if (last.get("review_state") == "pending" and last.get("review_action") == "addition"
                        and last.get("favorite_rule_version") == NOV24_VERSION
                        and last_reason.startswith("11/24 confirmation:")):
                    previous_signature, previous_stamps = json.loads(last_reason.split(":", 1)[1])
                    minimum = 15 if identity[0] == "nhl" else 10
                    if previous_signature == signature and 0 <= gap <= 30:
                        fresh = all(pd.Timestamp(new) > pd.Timestamp(old) for new, old in zip(stamps, previous_stamps))
                        if gap >= minimum and fresh:
                            extra_slates[slate] = identity[1]
                            review_records.append(_review_record(row, at, "applied", "addition", reason))
                            continue
                        _clear_favorite(result, index, "11/24 pathway awaits a fresh second confirmation.")
                        continue
            _clear_favorite(result, index, "11/24 pathway awaits a fresh second confirmation.")
            review_records.append(_review_record(row, at, "pending", "addition", reason))
            continue

        # A previously confirmed Favorite can legitimately recover after an
        # adverse retrace.  The raw qualification path intentionally rejects
        # high-churn markets, so recognize only a fully restored, stable move
        # back to the archived Favorite line.  This is not a new qualification:
        # it is constrained to the same supported side and still goes through
        # the normal two-capture visibility review below.
        recovered, recovery_reason = (
            (False, "") if (source is not None and cfb_spread_favorite(source)) or nba_public_favorite(source)
            or (nba_hold_legacy and identity[0] == "nba")
            else _confirmed_prior_favorite_recovery(row, source)
        )
        if recovered and not raw_favorite and not active:
            _restore_favorite(result, index, source)
            result.at[index, "favorite_reason"] = recovery_reason
            row = result.loc[index].copy()
            raw_favorite = True

        if minutes > review_minutes:
            # Additions and ordinary removals both need two consecutive
            # ten-minute pulls. This prevents one-cycle classification or
            # source-availability flicker from changing an established slate.
            if raw_favorite and not active:
                reason = "Awaiting a second consecutive qualifying scrape."
                if _review_confirmed(reviews, key, "addition", now, row):
                    review_records.append(_review_record(row, at, "applied", "addition", reason))
                else:
                    _clear_favorite(result, index, reason)
                    _set_review(result, index, "pending_addition", reason, at)
                    if not nba_public_favorite(row) or _nba_fresh_pending_source(reviews, key, row):
                        review_records.append(_review_record(row, at, "pending", "addition", reason))
            elif not raw_favorite and active:
                hard, _, reason = _material_removal(row, source)
                confirmed = _review_confirmed(reviews, key, "removal", now, source)
                if hard or confirmed:
                    if confirmed:
                        reason = _latest_review_reason(reviews, key) or reason
                    _clear_favorite(result, index, reason)
                    _set_review(result, index, "applied_removal", reason, at)
                    review_records.append(_review_record(row, at, "applied", "removal", reason, identity_row=source))
                else:
                    reason = (
                        str(row.get("favorite_reason", "")).strip()
                        if identity[0] in {"ufc", "nhl"}
                        else "Favorite failed one capture; awaiting a second consecutive nonqualifying scrape."
                    ) or "Favorite failed one capture; awaiting a second consecutive nonqualifying scrape."
                    _restore_favorite(result, index, source)
                    result.at[index, "favorite_reason"] = "Favorite held through a single nonqualifying capture."
                    _set_review(result, index, "pending_removal", reason, at)
                    review_records.append(_review_record(row, at, "pending", "removal", reason, identity_row=source))
            elif not raw_favorite and not active:
                _clear_pending_review(reviews, review_records, row, key, at)
            else:
                _clear_pending_review(reviews, review_records, row, key, at)
            continue

        if not 0 <= minutes <= review_minutes:
            continue

        if minutes <= lock_minutes:
            # Ordinary threshold flicker remains locked. A material addition
            # still needs its second confirmation; an adverse move suppresses
            # immediately.
            if identity[0] in {"ncaaf", "cfb", "ufc", "nhl"} and active and not raw_favorite:
                reason = (
                    row.get("_ncaaf_decision_reason", "")
                    or row.get("favorite_reason", "")
                    or f"{identity[0].upper()} Favorite failed mandatory T−{lock_minutes:g} requalification."
                )
                reason = f"Mandatory T−{lock_minutes:g} requalification failed: {reason}"
                _late_invalidate_favorite(result, index, reason, at)
                _set_review(result, index, "late_invalidated", reason, at)
                if not _review_already_recorded(reviews, key, "late_invalidated"):
                    review_records.append(_review_record(row, at, "late_invalidated", "removal", reason, identity_row=source)
                                          if source is not None and cfb_spread_favorite(source)
                                          else _review_record(row, at, "late_invalidated", "removal", reason))
                continue
            if active:
                _restore_favorite(result, index, source)
                result.at[index, "favorite_reason"] = "Favorite held pending confirmed material change."
            else:
                _clear_favorite(result, index, "Favorite addition requires confirmed material change.")
            if active:
                hard, destructive, reason = _material_removal(row, source)
                if hard or destructive:
                    _late_invalidate_favorite(result, index, reason, at)
                    _set_review(result, index, "late_invalidated", reason, at)
                    if not _review_already_recorded(reviews, key, "late_invalidated"):
                        review_records.append(_review_record(row, at, "late_invalidated", "removal", reason))
                    continue
            if raw_favorite == active:
                if raw_favorite and cfb_spread_favorite(row):
                    _restore_favorite(result, index, row)
                    result.at[index, "_visibility_restored"] = "false"
                    evidence = json.loads(row["favorite_supporting_evidence"])
                    evidence["closing_anchor_quote"] = _closing_review_quote(source)
                    result.at[index, "favorite_supporting_evidence"] = json.dumps(evidence, separators=(",", ":"))
                    result.at[index, "favorite_reason"] = "Official Favorite locked at T-20 after same-path requalification."
                    _set_review(result, index, "requalified_at_lock", result.at[index, "favorite_reason"], at)
                _clear_pending_review(reviews, review_records, row, key, at)
            elif raw_favorite:
                material, reason = _material_addition(row)
                if material and _review_confirmed(reviews, key, "addition", now, row):
                    _restore_favorite(result, index, row)
                    result.at[index, "_visibility_restored"] = "false"
                    _set_review(result, index, "applied_addition", reason, at)
                    review_records.append(_review_record(row, at, "applied", "addition", reason))
                elif material:
                    _set_review(result, index, "pending_addition", reason, at)
                    review_records.append(_review_record(row, at, "pending", "addition", reason))
            continue

        # T-40 through T-20: ordinary threshold flicker cannot change the
        # official slate. A material change needs two consecutive scrapes;
        # a confirmed opposite-side contradiction can remove immediately.
        if raw_favorite == active:
            _clear_pending_review(reviews, review_records, row, key, at)
            continue
        if raw_favorite:
            material, reason = _material_addition(row)
            if material and _review_confirmed(reviews, key, "addition", now, row):
                _set_review(result, index, "applied_addition", reason, at)
                review_records.append(_review_record(row, at, "applied", "addition", reason))
            else:
                _clear_favorite(result, index, "Closing-window addition withheld pending material confirmation.")
                if material:
                    _set_review(result, index, "pending_addition", reason, at)
                    review_records.append(_review_record(row, at, "pending", "addition", reason))
        else:
            hard, material, reason = _material_removal(row, source)
            confirmed = _review_confirmed(reviews, key, "removal", now, source)
            if confirmed:
                reason = _latest_review_reason(reviews, key) or reason
            if hard or confirmed:
                _clear_favorite(result, index, reason)
                _set_review(result, index, "applied_removal", reason, at)
                review_records.append(_review_record(row, at, "applied", "removal", reason, identity_row=source))
            else:
                _restore_favorite(result, index, source)
                result.at[index, "favorite_reason"] = "Favorite held through non-material closing-window change."
                if material:
                    _set_review(result, index, "pending_removal", reason, at)
                    review_records.append(_review_record(row, at, "pending", "removal", reason, identity_row=source))
    if "_nov24_added_support" in result or result["favorite_rule_version"].eq(NOV24_VERSION).any():
        picked = result["red_fox_favorite"].map(_truthy)
        added = result["favorite_rule_version"].eq(NOV24_VERSION)
        legacy_games = set(zip(result.loc[picked & ~added, "sport"], result.loc[picked & ~added, "game_id"]))
        for index, row in result.loc[picked & added].iterrows():
            if (row["sport"], row["game_id"]) in legacy_games:
                _late_invalidate_favorite(result, index, "11/24 pathway superseded by the existing Favorite pathway.", at)
    _write_review(review_path, reviews, review_records)
    if "_nov24_added_support" in result:
        provisional = result["_nov24_added_support"].eq(True) & ~result["red_fox_favorite"].map(_truthy)
        result.loc[provisional, "supported_side"] = ""
        result = result.drop(columns=["_nov24_added_support"])
    return result


def _history_for_key(history: pd.DataFrame, key: tuple[str, str, str], through: pd.Timestamp) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    mask = pd.Series(True, index=history.index)
    for column, value in zip(("sport", "game_id", "market_display"), key):
        if column not in history:
            return pd.DataFrame()
        mask &= history[column].astype(str).eq(value)
    scoped = history.loc[mask].copy()
    if scoped.empty:
        return scoped
    scoped["_at"] = pd.to_datetime(scoped["recorded_at"], errors="coerce", utc=True)
    return scoped.loc[scoped["_at"].notna() & (scoped["_at"] <= through)].sort_values("_at", kind="mergesort")


def _official_active(history: pd.DataFrame, key: tuple[str, str, str], through: pd.Timestamp) -> bool:
    scoped = _history_for_key(history, key, through)
    return bool(not scoped.empty and str(scoped.iloc[-1].get("favorite_state", "")) == "qualified")


def _read_review(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() and path.stat().st_size else pd.DataFrame(columns=REVIEW_COLUMNS)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=REVIEW_COLUMNS)


def _write_review(path: Path, history: pd.DataFrame, records: list[dict]) -> None:
    if not records:
        return
    updated = pd.concat([history, pd.DataFrame(records)], ignore_index=True, sort=False).reindex(columns=REVIEW_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    updated.to_csv(temporary, index=False)
    temporary.replace(path)


def _review_rows(history: pd.DataFrame, key: tuple[str, str, str]) -> pd.DataFrame:
    if history.empty:
        return history
    mask = pd.Series(True, index=history.index)
    for column, value in zip(("sport", "game_id", "market_display"), key):
        mask &= history.get(column, pd.Series("", index=history.index)).astype(str).eq(value)
    return history.loc[mask].sort_values("recorded_at", kind="mergesort")


def _favorite_review_identity(row: pd.Series | None) -> tuple[str, str]:
    if row is None:
        return "", ""
    return _side_identity(row.get("favorite_side", "")), str(row.get("favorite_rule_version", "")).strip()


def favorite_side_quote(row: pd.Series) -> str:
    """Read the quote for the identified team within this event-market row."""
    identity, _ = _favorite_review_identity(row)
    sides = _sides(row)
    matches = [side for side in sides if identity and _side_identity(side.get("flagged_side")) == identity]
    if len(matches) == 1:
        return str(matches[0].get("current_line") or "")
    if sides:
        return ""
    if row.get("market_display") == "SPREAD" or row.get("sport") in {"ufc", "nhl"}:
        try:
            evidence = json.loads(str(row.get("favorite_supporting_evidence") or "{}"))
        except (TypeError, ValueError):
            evidence = {}
        if isinstance(evidence, dict) and evidence.get("current_line"):
            return str(evidence["current_line"])
    leader = _side_identity(row.get("flagged_side", ""))
    if leader and leader != identity:
        return ""
    return str(row.get("current_line", ""))


def tracked_entry_quote(row: dict | pd.Series) -> tuple[str, str]:
    """Recover a legacy spread entry from its own first qualification evidence."""
    recorded = str(row.get("recorded_at", ""))
    if (row.get("market_display") != "SPREAD" or str(row.get("favorite_state", "")) != "qualified"
            or not row.get("favorite_side") or not recorded or recorded != row.get("first_qualified_at")):
        return "", ""
    try:
        evidence = json.loads(str(row.get("supporting_evidence") or "{}"))
    except (TypeError, ValueError):
        return "", ""
    if not isinstance(evidence, dict) or not evidence.get("current_line"):
        return "", ""
    return str(evidence["current_line"]), str(row.get("source_observed_at", ""))


def _review_source_observed_at(row: pd.Series, side_name: object) -> str:
    identity = _side_identity(side_name)
    side = next((item for item in _sides(row) if _side_identity(item.get("flagged_side")) == identity), {})
    return str(side.get("source_latest_at") or row.get("state_as_of_utc") or "").strip()


def _review_confirmed(history: pd.DataFrame, key: tuple[str, str, str], action: str, now: pd.Timestamp,
                      candidate: pd.Series | None) -> bool:
    scoped = _review_rows(history, key)
    identity = _favorite_review_identity(candidate)
    if scoped.empty or not all(identity):
        return False
    # Only the uninterrupted suffix for this exact decision can confirm it.
    matching = (
        scoped["review_state"].eq("pending") & scoped["review_action"].eq(action)
        & scoped.get("favorite_side", pd.Series("", index=scoped.index)).map(_side_identity).eq(identity[0])
        & scoped.get("favorite_rule_version", pd.Series("", index=scoped.index)).eq(identity[1])
    )
    interruptions = matching.reset_index(drop=True).index[~matching.to_numpy()]
    if len(interruptions):
        scoped = scoped.iloc[interruptions[-1] + 1:]
    if scoped.empty:
        return False
    pending_times = pd.to_datetime(scoped["recorded_at"], errors="coerce", utc=True)
    gaps = (now - pending_times).dt.total_seconds() / 60
    if action == "addition" and nba_public_favorite(candidate) and not _nba_fresh_pending_source(scoped, key, candidate):
        return False
    if key[0].lower() in {"ncaaf", "cfb"}:
        return bool(gaps.iloc[0] >= CONFIG.ncaaf_confirmation_min_gap_minutes)
    if key[0].lower() in {"ufc", "nhl"}:
        return bool(gaps.between(
            CONFIG.primary_moneyline_confirmation_minutes,
            CONFIG.primary_moneyline_confirmation_max_gap_minutes,
            inclusive="both",
        ).any())
    return bool(0 < gaps.iloc[-1] <= CONFIG.visibility_confirmation_gap_minutes)


def _nba_fresh_pending_source(reviews: pd.DataFrame, key: tuple[str, str, str], row: pd.Series) -> bool:
    scoped = _review_rows(reviews, key)
    if scoped.empty:
        return True
    last = scoped.iloc[-1]
    if (last.get("review_state") != "pending" or last.get("review_action") != "addition"
            or _favorite_review_identity(last) != _favorite_review_identity(row)):
        return True
    previous = pd.to_datetime(last.get("source_observed_at"), errors="coerce", utc=True)
    observed = pd.to_datetime(_review_source_observed_at(row, row.get("favorite_side")), errors="coerce", utc=True)
    return bool(pd.notna(observed) and (pd.isna(previous) or previous < observed))


def _review_already_recorded(history: pd.DataFrame, key: tuple[str, str, str], state: str) -> bool:
    scoped = _review_rows(history, key)
    return bool(not scoped.empty and str(scoped.iloc[-1].get("review_state", "")) == state)


def _latest_review_reason(history: pd.DataFrame, key: tuple[str, str, str]) -> str:
    scoped = _review_rows(history, key)
    return str(scoped.iloc[-1].get("reason", "")) if not scoped.empty else ""


def _review_record(row: pd.Series, at: str, state: str, action: str, reason: str,
                   *, identity_row: pd.Series | None = None) -> dict:
    candidate = row if identity_row is None else identity_row
    favorite_side = candidate.get("favorite_side", "")
    return {
        "recorded_at": at, "sport": row.get("sport", ""), "game_id": row.get("game_id", ""),
        "game": row.get("game", ""), "market_display": row.get("market_display", ""),
        "kickoff_iso": row.get("kickoff_iso", ""), "review_state": state,
        "review_action": action, "favorite_side": favorite_side,
        "current_line": row.get("current_line", ""), "reason": reason,
        "favorite_rule_version": candidate.get("favorite_rule_version", ""),
        "source_observed_at": _review_source_observed_at(row, favorite_side),
    }


def _clear_pending_review(history: pd.DataFrame, records: list[dict], row: pd.Series, key: tuple[str, str, str], at: str) -> None:
    scoped = _review_rows(history, key)
    if not scoped.empty and str(scoped.iloc[-1].get("review_state", "")) == "pending":
        records.append(_review_record(row, at, "cleared", str(scoped.iloc[-1].get("review_action", "")), "Pending review condition cleared."))


def _set_review(frame: pd.DataFrame, index: object, state: str, reason: str, at: str) -> None:
    frame.at[index, "favorite_review_state"] = state
    frame.at[index, "favorite_review_reason"] = reason
    frame.at[index, "favorite_review_at"] = at


def _material_addition(row: pd.Series) -> tuple[bool, str]:
    try:
        evidence = json.loads(str(row.get("favorite_supporting_evidence", "{}")))
    except (TypeError, ValueError, json.JSONDecodeError):
        evidence = {}
    bets = _number(evidence.get("bets_pct"))
    money = _number(evidence.get("money_pct"))
    if bets is None or money is None or bets > 35 or money > 35:
        return False, "Closing support is not sufficiently low (requires no more than 35% bets and money)."
    market = str(row.get("market_display", "")).upper()
    sport = str(row.get("sport", "")).lower()
    opened = _line_value(evidence.get("open_line"), market)
    current = _line_value(evidence.get("current_line"), market)
    move = abs(current - opened) if opened is not None and current is not None else 0
    if market == "MONEYLINE":
        material = move >= 15
        threshold = "15-cent Moneyline move"
    elif sport in {"nba", "ncaab", "cbb"}:
        material = move >= 1.5
        threshold = "1.5-point basketball move"
    else:
        crossed = set(re.findall(r"\d+(?:\.\d+)?", str(row.get("key_numbers_crossed", ""))))
        material = move >= 1.0 or bool(crossed & {"3", "7"})
        threshold = "one-point football move or key-number crossing"
    return material, f"Material closing addition: {threshold}; observed move {move:g}."


def _closing_review_quote(source: pd.Series | None) -> str:
    if source is None:
        return ""
    if cfb_spread_favorite(source):
        try:
            evidence = json.loads(str(source.get("favorite_supporting_evidence") or "{}"))
        except (TypeError, ValueError):
            evidence = {}
        if isinstance(evidence, dict) and evidence.get("closing_anchor_quote"):
            return str(evidence["closing_anchor_quote"])
    return favorite_side_quote(source)


def _material_removal(row: pd.Series, source: pd.Series | None) -> tuple[bool, bool, str]:
    if source is None:
        return False, False, "No prior frozen Favorite evidence was available."
    if nba_public_favorite(source) and _truthy(row.get("_nba_release_mismatch")):
        return True, True, "Hard removal: a confirmed cross-market contradiction appeared."
    if _truthy(row.get("return_toward_open")) and _truthy(row.get("active_worsening_reversal")):
        return (
            True,
            True,
            "Hard removal: the active game-week market fully returned toward its opener, erasing the qualifying move.",
        )
    favorite_identity = _side_identity(source.get("favorite_side", ""))
    supported_identity = _side_identity(row.get("supported_side", ""))
    if supported_identity and supported_identity != favorite_identity:
        return True, True, "Hard removal: the confirmed supported side flipped."
    if _truthy(row.get("cross_market_mismatch")) or _truthy(row.get("cross_market_opener_mismatch_verified")):
        return True, True, "Hard removal: a confirmed cross-market contradiction appeared."
    market = str(row.get("market_display", "")).upper()
    side = next((item for item in _sides(row) if _side_identity(item.get("flagged_side")) == favorite_identity), None)
    if side is None:
        return False, False, "The Favorite side was temporarily unavailable on one scrape."
    sport = str(row.get("sport", "")).lower()
    if sport in {"ufc", "nhl"}:
        context = _parts(side.get("context_chips")) | _parts(side.get("anomaly_chips"))
        hard_risks = {"Feed Risk"}
        if sport == "ufc":
            hard_risks |= {"Bout Suspended", "Opponent Change", "Weight Miss", "Weigh-In Risk"}
        else:
            hard_risks |= {"Goalie Change", "Goalie Unconfirmed", "Goalie Risk"}
        risks = sorted(context & hard_risks)
        if risks:
            return True, True, f"Hard removal: blocking {sport.upper()} risk state {', '.join(risks)}."
        if str(side.get("data_badge", "")) != "Clean":
            return True, True, f"Hard removal: {sport.upper()} market data is no longer Clean."
    previous_line = _closing_review_quote(source)
    previous = _line_value(previous_line, market)
    current = _line_value(side.get("current_line", ""), market)
    if market == "MONEYLINE" and sport in {"ufc", "nhl"}:
        previous_probability = _american_implied_probability(previous)
        current_probability = _american_implied_probability(current)
        against = (
            previous_probability - current_probability
            if previous_probability is not None and current_probability is not None
            else 0
        )
        threshold = (
            CONFIG.ufc_adverse_probability_points
            if sport == "ufc"
            else CONFIG.nhl_adverse_probability_points
        )
        material = against >= threshold
        return (
            False,
            material,
            f"Material closing removal: implied probability moved {against:g} points against "
            f"the Favorite (threshold {threshold:g}).",
        )
    against = (current - previous) if previous is not None and current is not None else 0
    threshold = 15 if market == "MONEYLINE" else (
        1.5 if sport in {"nba", "ncaab", "cbb"}
        else CONFIG.ncaaf_adverse_move_threshold if sport in {"ncaaf", "cfb"}
        else 1.0
    )
    material = against >= threshold
    return False, material, f"Material closing removal: market moved {against:g} against the Favorite (threshold {threshold:g})."


def _american_implied_probability(odds: float | None) -> float | None:
    if odds is None or odds == 0:
        return None
    return 100.0 * (-odds / (-odds + 100.0) if odds < 0 else 100.0 / (odds + 100.0))


def _confirmed_prior_favorite_recovery(row: pd.Series, source: pd.Series | None) -> tuple[bool, str]:
    """Recognize a durable return to a previously archived Favorite state."""
    if source is None or str(row.get("market_display", "")).upper() == "TOTAL":
        return False, ""
    favorite_identity = _side_identity(source.get("favorite_side", ""))
    supported_identity = _side_identity(row.get("supported_side", ""))
    if not favorite_identity or supported_identity != favorite_identity:
        return False, ""
    if _truthy(row.get("cross_market_mismatch")) or _truthy(row.get("cross_market_opener_mismatch_verified")):
        return False, ""
    side = next((item for item in _sides(row) if _side_identity(item.get("flagged_side")) == favorite_identity), None)
    if side is None or str(side.get("data_badge", "")) != "Clean":
        return False, ""
    bets, money = _number(side.get("bets_pct")), _number(side.get("money_pct"))
    if bets is None or money is None or bets > 35 or money > 35:
        return False, ""
    if (
        str(side.get("reaction", "")) != "Contrarian"
        or str(side.get("response_direction", "")).upper() != "TOWARD"
        or not _truthy(side.get("kpi_eligible"))
        or _truthy(side.get("return_toward_open"))
        or _truthy(side.get("active_worsening_reversal"))
        or str(side.get("path", "")) != "Whipsaw"
        or (_number(side.get("whipsaw_retention_ratio")) or 0) < 0.95
        or not _truthy(side.get("whipsaw_confirmation_ready"))
        or (_number(side.get("whipsaw_confirmation_minutes")) or 0) < CONFIG.partial_whipsaw_confirmation_minutes
    ):
        return False, ""
    market = str(row.get("market_display", "")).upper()
    archived = _line_value(favorite_side_quote(source), market)
    current = _line_value(side.get("current_line", ""), market)
    if archived is None or current is None or current > archived:
        return False, ""
    return True, "Previously confirmed Favorite restored after the qualifying market move fully returned and held."


def _freshness_failure(row: pd.Series | None, source: pd.Series | None, now: pd.Timestamp) -> str:
    candidate_side = str((source.get("favorite_side", "") if source is not None else "") or (row.get("favorite_side", "") if row is not None else "") or (row.get("supported_side", "") if row is not None else "")).strip()
    evidence_row = row if row is not None else source
    if evidence_row is None or not candidate_side:
        return ""
    side = next((item for item in _sides(evidence_row) if _side_identity(item.get("flagged_side")) == _side_identity(candidate_side)), None)
    observed = str((side or {}).get("source_latest_at", "") or evidence_row.get("source_latest_at", "")).strip()
    source_at = pd.to_datetime(observed, errors="coerce", utc=True)
    if pd.isna(source_at):
        return ""
    whipsaw = (
        str((side or {}).get("path", "")) == "Whipsaw"
        or _truthy((side or {}).get("whipsaw_recovered"))
        or str(evidence_row.get("favorite_whipsaw_state", "")) in {"partial_retrace_intact", "recovered"}
    )
    maximum = CONFIG.whipsaw_freshness_minutes if whipsaw else CONFIG.normal_freshness_minutes
    age = (now - source_at).total_seconds() / 60.0
    if age > maximum:
        return f"Favorite unavailable: latest verified market state is {age:.1f} minutes old (maximum {maximum})."
    return ""


def _archived_row(archived: pd.DataFrame, key: tuple[str, str, str]) -> pd.Series | None:
    if archived.empty:
        return None
    mask = pd.Series(True, index=archived.index)
    for column, value in zip(("sport", "game_id", "market_display"), key):
        if column not in archived:
            return None
        mask &= archived[column].astype(str).eq(value)
    matches = archived.loc[mask]
    return matches.iloc[-1] if not matches.empty else None


def _restore_favorite(frame: pd.DataFrame, index: object, source: pd.Series | None) -> None:
    if source is None:
        return
    for column in FAVORITE_COLUMNS:
        frame.at[index, column] = source.get(column, "")
    frame.at[index, "red_fox_favorite"] = "true"
    frame.at[index, "favorite_state"] = "qualified"
    frame.at[index, "_visibility_restored"] = "true"


def _clear_favorite(frame: pd.DataFrame, index: object, reason: str) -> None:
    for column in FAVORITE_COLUMNS:
        frame.at[index, column] = ""
    frame.at[index, "red_fox_favorite"] = "false"
    frame.at[index, "favorite_rule_version"] = _favorite_rule_version(frame.at[index, "sport"])
    frame.at[index, "favorite_state"] = "not_qualified"
    frame.at[index, "favorite_reason"] = reason


def _late_invalidate_favorite(frame: pd.DataFrame, index: object, reason: str, at: str) -> None:
    _clear_favorite(frame, index, reason)
    frame.at[index, "favorite_originally_qualified"] = "true"
    frame.at[index, "favorite_late_invalidated"] = "true"
    frame.at[index, "favorite_late_invalidated_at"] = at
    frame.at[index, "favorite_late_invalidated_reason"] = reason


def _favorite_rule_version(sport: object) -> str:
    normalized = str(sport).lower()
    if normalized in {"ncaaf", "cfb"}:
        return CONFIG.ncaaf_version
    if normalized in {"nba", "ncaab", "cbb"}:
        return CONFIG.basketball_version
    if normalized == "ufc":
        return CONFIG.ufc_version
    if normalized == "nhl":
        return CONFIG.nhl_version
    return CONFIG.version


def _legacy_rejection_reason(row: pd.Series, side: dict | None, game_rows: pd.DataFrame | None) -> str:
    """Return the first failed legacy gate for NCAAF shadow/audit review."""
    if side is None:
        return "Legacy Favorite rejected: no confirmed Supported Side matched this market."
    if str(side.get("data_badge", "")) != "Clean":
        return "Legacy Favorite rejected: market data was not Clean."
    observations = int(_number(side.get("observation_count")) or 0)
    if observations < CONFIG.minimum_observations:
        return f"Legacy Favorite rejected: {observations} observations; requires at least {CONFIG.minimum_observations}."
    context = _parts(side.get("context_chips"))
    risk = sorted({"Market Lag", "Feed Risk", "Split Risk", "Split Cap"} & context)
    if risk:
        return f"Legacy Favorite rejected: blocking data/risk state {', '.join(risk)}."
    if _truthy(side.get("active_worsening_reversal")):
        return "Legacy Favorite rejected: active worsening reversal."
    market = str(row.get("market_display", "")).upper()
    sport = str(row.get("sport", "")).lower()
    value = _line_value(side.get("current_line"), market)
    if value is None or not _number_is_eligible(sport, market, value, side, game_rows):
        return "Legacy Favorite rejected: candidate line/price is outside the eligible range."
    target = _side_identity(side.get("flagged_side"))
    other = next((item for item in _sides(row) if _side_identity(item.get("flagged_side")) != target), None)
    if other is None or not _pathway(side, other, sport, market):
        return "Legacy Favorite rejected: split, direction, reaction, or resistance pathway did not qualify."
    if not _matches_confirmed_supported_side(row, side):
        return "Legacy Favorite rejected: candidate was not the confirmed Supported Side."
    if market == "SPREAD" and _corresponding_moneyline_state(side, game_rows) == "contradiction":
        return "Legacy Favorite rejected: corresponding Moneyline contradicted the spread candidate."
    return "Legacy Favorite rejected: one or more legacy quality gates did not qualify."


def _fresh_nov24_side(side: dict, now: pd.Timestamp) -> bool:
    observed = pd.to_datetime(side.get("source_latest_at"), errors="coerce", utc=True)
    return bool(pd.notna(observed) and pd.Timedelta(0) <= now - observed <= pd.Timedelta(minutes=12))


def _nov24_decision(row, game_rows, at, goalie_cache):
    sport = str(row.get("sport", "")).lower()
    market = str(row.get("market_display", "")).upper()
    now = pd.Timestamp(at)
    kickoff = pd.to_datetime(row.get("kickoff_iso"), errors="coerce", utc=True)
    if pd.isna(kickoff) or not pd.Timedelta(0) < kickoff - now <= pd.Timedelta(hours=24):
        return None
    basketball = sport in {"nba", "ncaab"} and market == "SPREAD"
    if not basketball and (sport != "nhl" or market != "MONEYLINE"):
        return None
    goalies = None
    if sport == "nhl":
        from favorite_nhl_goalies import confirmed_goalies
        goalies = confirmed_goalies(goalie_cache, str(row.get("game", "")), kickoff, now)
        if not goalies:
            return None
    sides = _sides(row)
    if len(sides) != 2:
        return None
    supported = _side_identity(row.get("supported_side", ""))
    for side, other in ((sides[0], sides[1]), (sides[1], sides[0])):
        name = str(side.get("flagged_side", ""))
        identity = _side_identity(name)
        if not identity or identity == _side_identity(other.get("flagged_side", "")):
            continue
        if (supported and supported != identity) or str(other.get("reaction", "")) in {"Contrarian", "Follow"}:
            continue
        if any(
            str(item.get("reaction", "")) == "Freeze"
            and _truthy(item.get("kpi_eligible"))
            and _side_identity(item.get("action_side", "")) != identity
            for item in sides
        ):
            continue
        bets, money = _number(side.get("bets_pct")), _number(side.get("money_pct"))
        value = _line_value(side.get("current_line"), market)
        if (bets is None or money is None or not 0 <= bets <= 45 or value is None
                or not _number_is_eligible(sport, market, value, side, game_rows)
                or not _base_quality(side) or not _fresh_nov24_side(side, now)):
            continue
        sources = [str(side["source_latest_at"])]
        if basketball:
            opened = _line_value(side.get("open_line"), market)
            move = opened - value if opened is not None else 0
            if not (value > 0 and 0 <= money <= 45 and 1 <= move < CONFIG.basketball_spread_move_min):
                continue
            moneyline = _matching_market_side(game_rows, "MONEYLINE", name)
            ml_row = _matching_market_row(game_rows, "MONEYLINE", name)
            if (not moneyline or ml_row is None or not _base_quality(moneyline)
                    or not _fresh_nov24_side(moneyline, now)
                    or _basketball_moneyline_alignment(moneyline) != "confirmation"
                    or _side_identity(ml_row.get("supported_side", "")) not in {"", identity}
                    or _truthy(ml_row.get("cross_market_mismatch"))
                    or _truthy(ml_row.get("cross_market_opener_mismatch_verified"))):
                continue
            if sport == "nba" and (_number(moneyline.get("price_move_pct")) or 0) < CONFIG.meaningful_moneyline_price_move:
                continue
            pathway, cross_state = "low_support_contrarian", "confirmation"
            sources.append(str(moneyline["source_latest_at"]))
            reason = "11/24 one-point spread exception: fresh same-side spread and moneyline movement."
        else:
            if not bets + 15 <= money <= 100 or not _primary_moneyline_sport_gate(row, side, "low_support_contrarian", at):
                continue
            pathway, cross_state = "low_ticket_high_money_contrarian", "not_applicable"
            reason = "11/24 NHL low-ticket, higher-money continuation with both starting goalies confirmed."
        return {
            "nov24": True, "side": name, "pathway": pathway,
            "cross_market_state": cross_state,
            "whipsaw_state": "recovered" if _truthy(side.get("whipsaw_recovered")) else "none",
            "market_read": str(side.get("anomaly_chips") or side.get("reaction", "")),
            "reason": reason,
            "evidence": {
                "reaction": str(side.get("reaction", "")), "path": str(side.get("path", "")),
                "bets_pct": bets, "money_pct": money, "open_line": side.get("open_line", ""),
                "current_line": side.get("current_line", ""), "confirmed_goalies": list(goalies or ()),
                "cross_market": cross_state,
                "confirmation_sources": sources,
                "observation_count": int(_number(side.get("observation_count")) or 0),
            },
        }
    return None


def _qualify_market(
    row: pd.Series,
    game_rows: pd.DataFrame | None,
    *,
    enforce_ncaaf_policy: bool = True,
    at: str | None = None,
    failures: dict[str, str] | None = None,
) -> dict | None:
    sport = str(row.get("sport", "")).lower()
    market = str(row.get("market_display", "")).upper()
    if failures is None:
        failures = {}
    if market == "TOTAL":
        failures[""] = "Market type TOTAL is ineligible."
        return None
    if market == "SPREAD" and sport not in CONFIG.spread_sports:
        failures[""] = "Sport does not have an official spread pathway."
        return None
    if market == "MONEYLINE" and sport not in CONFIG.primary_moneyline_sports | CONFIG.secondary_moneyline_sports:
        failures[""] = "Sport does not have an official moneyline pathway."
        return None
    sides = _sides(row)
    if len(sides) != 2:
        failures[""] = "Market lacks an exact two-sided evidence pair."
        return None
    if not str(row.get("supported_side") or "").strip():
        failures[""] = "Market lacks a confirmed Supported Side."

    candidates = []
    for side in sides:
        identity = _side_identity(side.get("flagged_side", ""))
        other = next((item for item in sides if item is not side), None)
        if other is None:
            continue
        base_quality = _base_quality(side)
        if sport in {"ncaaf", "cfb"} and _intact_ncaaf_partial_whipsaw(side):
            base_quality = True
        if not base_quality:
            failures[identity] = _quality_failure(side)
            continue
        value = _line_value(side.get("current_line"), market)
        if value is None or not _number_is_eligible(sport, market, value, side, game_rows):
            failures[identity] = "The recorded line, price, or paired-market price eligibility gate failed."
            continue
        pathway = _pathway(side, other, sport, market)
        if not pathway:
            failures[identity] = "No Favorite pathway matched the current split, reaction and movement evidence."
            continue
        if sport in {"ufc", "nhl"} and not _primary_moneyline_sport_gate(
            row, side, pathway, at,
        ):
            failures[identity] = _primary_moneyline_rejection_reason(row, game_rows, at)
            continue
        # Favorite is a narrower designation layered over the published
        # Market Read.  Never publish a Favorite when that same side is not
        # the market's authoritative confirmed supported side; otherwise the
        # red badge can appear with no corresponding green row (or on the
        # opposite row).  Do not infer support here from movement/context --
        # ``supported_side`` is resolved upstream by the Market Read engine.
        if not _matches_confirmed_supported_side(row, side):
            failures[identity] = "Candidate did not match the exact confirmed Supported Side."
            continue
        cross_state = "not_applicable"
        if market == "SPREAD":
            cross_state = _corresponding_moneyline_state(side, game_rows)
            if cross_state == "contradiction":
                failures[identity] = "Matching moneyline movement contradicted the spread candidate."
                continue
        basketball_gate = {}
        if sport in {"nba", "ncaab", "cbb"}:
            basketball_gate, cross_state = _basketball_gate_for_market(
                row, side, game_rows,
            )
            if not basketball_gate["passed"]:
                failures[identity] = basketball_gate["reason"]
                continue
        ncaaf_gate = {}
        if enforce_ncaaf_policy and sport in {"ncaaf", "cfb"}:
            ncaaf_gate, cross_state = _ncaaf_gate_for_market(row, side, game_rows)
            if not ncaaf_gate["passed"]:
                failures[identity] = ncaaf_gate["reason"]
                continue
            if pathway == "low_ticket_high_money_contrarian" and cross_state != "confirmation":
                failures[identity] = "CFB high-handle pathway lacked confirming moneyline movement."
                continue
        candidates.append((side, pathway, cross_state, ncaaf_gate, basketball_gate))
    if not candidates:
        return None
    priority = {
        "low_support_contrarian": 0,
        "low_ticket_high_money_contrarian": 1,
        "low_support_freeze": 2,
        "moderate_support_follow": 3,
    }
    side, pathway, cross_state, ncaaf_gate, basketball_gate = min(
        candidates, key=lambda item: priority[item[1]],
    )
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
    if ncaaf_gate:
        evidence.update({
            "ncaaf_move": ncaaf_gate["move"],
            "ncaaf_move_gate_passed": ncaaf_gate["move_gate_passed"],
            "ncaaf_market_move_status": ncaaf_gate["market_move_status"],
            "open_observed_at": str(side.get("open_observed_at", "")),
        })
    if basketball_gate:
        evidence.update({
            "basketball_spread": basketball_gate["spread"],
            "basketball_spread_move": basketball_gate["spread_move"],
            "basketball_moneyline_move": basketball_gate["moneyline_move"],
            "basketball_expression": basketball_gate["expression"],
        })
    reason = {
        "low_support_contrarian": "Lower-supported side received a meaningful, intact move toward it.",
        "low_ticket_high_money_contrarian": (
            "Low-ticket side retained concentrated money support while the confirmed spread and moneyline moves remained intact."
        ),
        "low_support_freeze": "Lower-supported side is protected by the engine's actionable, persistent resistance state.",
        "moderate_support_follow": "Moderately supported side received a meaningful, intact confirming move.",
    }[pathway]
    if ncaaf_gate:
        reason = f"{reason} {ncaaf_gate['reason']}"
    if basketball_gate:
        reason = f"{reason} {basketball_gate['reason']}"
    return {
        "side": str(side.get("flagged_side", "")), "pathway": pathway,
        "cross_market_state": cross_state, "whipsaw_state": whipsaw_state,
        "evidence": evidence, "reason": reason,
        "market_read": str(side.get("anomaly_chips") or side.get("reaction", "")),
        "ncaaf_gate": ncaaf_gate,
        "basketball_gate": basketball_gate,
    }


def _basketball_gate_for_market(
    row: pd.Series,
    side: dict,
    game_rows: pd.DataFrame | None,
) -> tuple[dict, str]:
    """Select one NBA/NCAAB expression and require cross-market support.

    Point receivers are expressed only through the spread. Teams laying points
    are expressed only through the moneyline. The normal confirmation is a
    1.5-point spread move plus clean, aligned moneyline evidence toward the same
    team; the confirming moneyline does not have to independently clear a
    second movement threshold. A team laying no more than 1.5 points may still
    qualify without a spread move, but that exception does require meaningful
    moneyline movement.
    """
    market = str(row.get("market_display", "")).upper()
    spread = side if market == "SPREAD" else _matching_market_side(
        game_rows, "SPREAD", side.get("flagged_side"),
    )
    moneyline = side if market == "MONEYLINE" else _matching_market_side(
        game_rows, "MONEYLINE", side.get("flagged_side"),
    )
    moneyline_row = row if market == "MONEYLINE" else _matching_market_row(
        game_rows, "MONEYLINE", side.get("flagged_side"),
    )
    spread_value = _line_value((spread or {}).get("current_line"), "SPREAD")
    spread_move = _number((spread or {}).get("line_move_abs")) or 0.0
    moneyline_move = _number((moneyline or {}).get("price_move_pct")) or 0.0
    moneyline_alignment = _basketball_moneyline_alignment(moneyline)
    moneyline_support = moneyline_alignment == "confirmation"
    meaningful_moneyline_support = bool(
        moneyline_support
        and moneyline_move >= CONFIG.meaningful_moneyline_price_move
    )
    cross_state = moneyline_alignment

    def outcome(passed: bool, expression: str, reason: str) -> tuple[dict, str]:
        return ({
            "passed": passed,
            "spread": spread_value,
            "spread_move": spread_move,
            "moneyline_move": moneyline_move,
            "expression": expression,
            "reason": reason,
        }, cross_state)

    if spread_value is None:
        return outcome(False, "none", "Basketball Favorite withheld: a matching spread is required.")
    if spread_value > 0:
        if market != "SPREAD":
            return outcome(
                False, "spread",
                "Basketball Favorite is expressed on the positive spread because the supported team is receiving points.",
            )
        if spread_move < CONFIG.basketball_spread_move_min:
            return outcome(
                False, "spread",
                f"Basketball Favorite withheld: requires at least a {CONFIG.basketball_spread_move_min:g}-point spread move.",
            )
        if not moneyline_support:
            return outcome(
                False, "spread",
                "Basketball Favorite withheld: the qualifying spread move requires confirming moneyline support.",
            )
        return outcome(
            True, "spread",
            "Basketball Favorite confirmed by a 1.5-point-or-greater spread move with aligned moneyline support; the team is receiving points, so the spread is the sole recorded Favorite.",
        )

    moneyline_value = _line_value((moneyline or {}).get("current_line"), "MONEYLINE")
    eligible_moneyline_expression = bool(
        moneyline_value is not None
        and CONFIG.moneyline_min <= moneyline_value < 0
        and abs(spread_value) <= CONFIG.secondary_moneyline_spread_max
        and moneyline_row is not None
        and _side_identity(moneyline_row.get("supported_side"))
        == _side_identity(moneyline.get("flagged_side"))
    )
    if not eligible_moneyline_expression:
        return outcome(
            False, "moneyline",
            "Basketball Favorite withheld: a team laying points requires an eligible matching moneyline from -165 up to negative money and a spread no larger than four points.",
        )
    short_favorite = abs(spread_value) <= CONFIG.basketball_short_favorite_spread_max
    if market == "MONEYLINE" and short_favorite and spread_move < CONFIG.basketball_spread_move_min:
        if not meaningful_moneyline_support:
            return outcome(
                False, "moneyline",
                f"Basketball Favorite withheld: the no-spread-move exception requires at least {CONFIG.meaningful_moneyline_price_move:g} implied-probability points of moneyline support.",
            )
        return outcome(
            True, "moneyline",
            "Basketball Favorite confirmed by meaningful moneyline support at pick'em through -1.5; no spread movement is required, and the moneyline is the sole recorded Favorite.",
        )
    if spread_move < CONFIG.basketball_spread_move_min:
        return outcome(
            False, "moneyline",
            f"Basketball Favorite withheld: requires at least a {CONFIG.basketball_spread_move_min:g}-point spread move with aligned moneyline support.",
        )
    if not moneyline_support:
        return outcome(
            False, "moneyline",
            "Basketball Favorite withheld: the qualifying spread move requires clean moneyline movement aligned toward the same team.",
        )
    return outcome(
        True, "moneyline",
        "Basketball Favorite confirmed by a 1.5-point-or-greater spread move with aligned moneyline support; the team is laying points, so the moneyline is the sole recorded Favorite.",
    )


def _basketball_moneyline_alignment(moneyline: dict | None) -> str:
    """Return reliable directional support without imposing a second threshold."""
    if not moneyline:
        return "unavailable"
    if (
        str(moneyline.get("data_badge", "")) != "Clean"
        or int(_number(moneyline.get("observation_count")) or 0) < CONFIG.minimum_observations
    ):
        return "unavailable"
    direction = str(moneyline.get("response_direction", "")).upper()
    move = _number(moneyline.get("price_move_pct")) or 0.0
    if direction == "TOWARD" and move > 0:
        return "confirmation"
    if direction == "AGAINST" and move >= CONFIG.meaningful_moneyline_price_move:
        return "contradiction"
    return "neutral"


def _primary_moneyline_sport_gate(
    row: pd.Series,
    side: dict,
    pathway: str,
    at: str | None,
) -> bool:
    """Apply forward-only strict qualification to UFC and NHL moneylines."""
    sport = str(row.get("sport", "")).lower()
    if sport not in {"ufc", "nhl"}:
        return True
    if str(row.get("market_display", "")).upper() != "MONEYLINE":
        return False
    # Split-only resistance and moderate-follow signals remain useful board
    # context, but are not official UFC/NHL Favorites without a low-support,
    # movement-backed Contrarian read.
    if pathway != "low_support_contrarian":
        return False
    threshold = CONFIG.ufc_market_move_min if sport == "ufc" else CONFIG.nhl_market_move_min
    if (
        str(side.get("response_direction", "")).upper() != "TOWARD"
        or (_number(side.get("price_move_pct")) or 0) < threshold
    ):
        return False
    context = _parts(side.get("context_chips")) | _parts(side.get("anomaly_chips"))
    hard_risks = (
        {"Bout Suspended", "Opponent Change", "Weight Miss", "Weigh-In Risk"}
        if sport == "ufc"
        else {"Goalie Change", "Goalie Unconfirmed", "Goalie Risk"}
    )
    if context & hard_risks:
        return False
    kickoff = pd.to_datetime(row.get("kickoff_iso", ""), errors="coerce", utc=True)
    captured = pd.to_datetime(at, errors="coerce", utc=True)
    if pd.isna(kickoff) or pd.isna(captured):
        return False
    hours = (kickoff - captured).total_seconds() / 3600.0
    active_window = CONFIG.ufc_active_window_hours if sport == "ufc" else CONFIG.nhl_active_window_hours
    if hours < 0 or hours > active_window:
        return False
    recovered = _truthy(side.get("whipsaw_recovered")) or "Whipsaw Recovered" in context
    ever_whipsaw = recovered or str(side.get("path", "")) == "Whipsaw"
    if ever_whipsaw:
        if not recovered:
            return False
        if (_number(side.get("whipsaw_retention_ratio")) or 0) < CONFIG.primary_moneyline_whipsaw_min_retention:
            return False
        changes = int(_number(side.get("material_direction_changes")) or 0)
        if changes > CONFIG.primary_moneyline_whipsaw_max_direction_changes:
            return False
    return True


def _primary_moneyline_rejection_reason(
    row: pd.Series,
    game_rows: pd.DataFrame | None,
    at: str | None,
) -> str:
    """Return the exact first failed UFC/NHL forward-policy gate."""
    sport = str(row.get("sport", "")).lower()
    label = sport.upper()
    supported = str(row.get("supported_side", "")).strip()
    side = _matching_market_side(game_rows, "MONEYLINE", supported) if supported else None
    if side is None:
        return f"{label} Favorite withheld: no exact currently confirmed Supported Side."
    other = next(
        (
            item for item in _sides(row)
            if _side_identity(item.get("flagged_side")) != _side_identity(side.get("flagged_side"))
        ),
        {},
    )
    pathway = _pathway(
        side,
        other,
        sport,
        "MONEYLINE",
    )
    if pathway != "low_support_contrarian":
        return f"{label} Favorite withheld: requires the low-support, movement-backed Contrarian pathway."
    threshold = CONFIG.ufc_market_move_min if sport == "ufc" else CONFIG.nhl_market_move_min
    if str(side.get("response_direction", "")).upper() != "TOWARD":
        return f"{label} Favorite withheld: movement is not toward the candidate."
    if (_number(side.get("price_move_pct")) or 0) < threshold:
        return f"{label} Favorite withheld: requires at least {threshold:g} implied-probability points toward the candidate."
    context = _parts(side.get("context_chips")) | _parts(side.get("anomaly_chips"))
    hard_risks = (
        {"Bout Suspended", "Opponent Change", "Weight Miss", "Weigh-In Risk"}
        if sport == "ufc"
        else {"Goalie Change", "Goalie Unconfirmed", "Goalie Risk"}
    )
    risks = sorted(context & hard_risks)
    if risks:
        return f"{label} Favorite withheld: blocking risk state {', '.join(risks)}."
    kickoff = pd.to_datetime(row.get("kickoff_iso", ""), errors="coerce", utc=True)
    captured = pd.to_datetime(at, errors="coerce", utc=True)
    active_window = CONFIG.ufc_active_window_hours if sport == "ufc" else CONFIG.nhl_active_window_hours
    if pd.isna(kickoff) or pd.isna(captured):
        return f"{label} Favorite withheld: a verified scheduled start is required."
    hours = (kickoff - captured).total_seconds() / 3600.0
    if hours < 0:
        return f"{label} Favorite withheld: the scheduled start has passed."
    if hours > active_window:
        return f"{label} Favorite withheld: official status begins inside T-{active_window:g} hours."
    recovered = _truthy(side.get("whipsaw_recovered")) or "Whipsaw Recovered" in context
    if str(side.get("path", "")) == "Whipsaw" and not recovered:
        return f"{label} Favorite withheld: unresolved whipsaw movement."
    if recovered and (_number(side.get("whipsaw_retention_ratio")) or 0) < CONFIG.primary_moneyline_whipsaw_min_retention:
        return f"{label} Favorite withheld: recovered whipsaw retained less than 80% of its maximum move."
    if recovered and int(_number(side.get("material_direction_changes")) or 0) > CONFIG.primary_moneyline_whipsaw_max_direction_changes:
        return f"{label} Favorite withheld: recovered whipsaw exceeded two material direction changes."
    return f"{label} Favorite withheld: current strict qualification gates were not satisfied."


def _cfb_held_spread(side: dict, other: dict, history: pd.DataFrame, opened: float,
                     opener: pd.Timestamp, observed: pd.Timestamp) -> bool:
    """Match the research hold: positive throughout a 30-minute lookback."""
    if history.empty:
        return False
    wanted = _side_identity(side["flagged_side"])
    opponent = _side_identity(other["flagged_side"])
    times = pd.to_datetime(history["timestamp"], errors="coerce", utc=True, format="mixed")
    eligible = history.loc[times.between(max(opener, observed - pd.Timedelta(hours=1)), observed)]
    pairs = {}
    conflicts = set()
    for ts, name, quote in zip(times.loc[eligible.index], eligible["side"], eligible["current_line"]):
        identity = _side_identity(name)
        if identity not in {wanted, opponent}:
            continue
        pair = pairs.setdefault(ts, {})
        value = _line_value(quote, "SPREAD")
        if identity in pair and pair[identity] != value:
            conflicts.add(ts)
        pair[identity] = value
    captures = sorted((ts, pair[wanted]) for ts, pair in pairs.items()
                      if ts not in conflicts and pair.keys() == {wanted, opponent}
                      and all(value is not None for value in pair.values())
                      and abs(pair[wanted] + pair[opponent]) < .001)
    anchor = max((ts for ts, _ in captures if ts <= observed - pd.Timedelta(minutes=30)), default=None)
    return (anchor is not None and captures[-1] == (observed, _line_value(side["current_line"], "SPREAD"))
            and all(opened > value for ts, value in captures if ts >= anchor))


def _cfb_spread_decision(row: pd.Series, game: pd.DataFrame, history: pd.DataFrame,
                         now: pd.Timestamp, allowed: frozenset[str], prior_side: str = "") -> dict | None:
    if not allowed or str(row.get("market_display", "")).upper() != "SPREAD":
        return None
    kickoff = pd.to_datetime(row.get("kickoff_iso"), errors="coerce", utc=True)
    if pd.isna(kickoff) or kickoff <= now:
        return None
    if any(game.get(key, pd.Series(dtype=str)).map(_truthy).any()
           for key in ("cross_market_mismatch", "cross_market_opener_mismatch_verified")):
        return None
    sides = _sides(row)
    if len(sides) != 2 or len({_side_identity(side.get("flagged_side")) for side in sides} - {""}) != 2:
        return None
    supported = _side_identity(row.get("supported_side"))
    candidates = []
    for side, other in (sides, sides[::-1]):
        name = str(side["flagged_side"])
        identity = _side_identity(name)
        if (supported and supported != identity) or (prior_side and prior_side != identity):
            continue
        bets = _number(side.get("bets_pct"))
        money = _number(side.get("money_pct"))
        if bets is None or money is None:
            continue
        public = CFB_PUBLIC_FOLLOW in allowed and bets >= 70 and money >= 55
        high = CFB_HIGH_MONEY in allowed and bets <= 35 and money - bets >= 15
        if not (public or high):
            continue
        current = _line_value(side.get("current_line"), "SPREAD")
        opened = _line_value(side.get("open_line"), "SPREAD")
        if current is None or opened is None or opened - current < 1.5:
            continue
        if not 1 <= abs(current) <= (8 if public else 21):
            continue
        # Published two-side histories start at the same paired capture. The
        # market row retains its opener clock even when the side JSON omits it.
        open_at = pd.to_datetime(side.get("open_observed_at") or row.get("open_observed_at"), errors="coerce", utc=True)
        source_at = pd.to_datetime(side.get("source_latest_at"), errors="coerce", utc=True)
        if not kickoff - pd.Timedelta(days=CONFIG.ncaaf_active_market_window_days) <= open_at <= source_at <= now:
            continue
        quality = _quality_failure(side)
        intact = _intact_ncaaf_partial_whipsaw(side)
        if quality and not (quality.startswith("Unrecovered Whipsaw") and intact):
            continue
        if str(side.get("path", "")) == "Whipsaw" and not (
            _truthy(side.get("whipsaw_recovered")) or "Whipsaw Recovered" in _parts(side.get("context_chips")) or intact
        ):
            continue
        selected = row.copy()
        selected["favorite_side"] = name
        if _freshness_failure(selected, None, now):
            continue
        cross = _corresponding_moneyline_state(side, game)
        if cross == "contradiction" or (public and cross != "confirmation"):
            continue
        if high and not _cfb_held_spread(side, other, history, opened, open_at, source_at):
            continue
        candidates.append({
            "side": name, "pathway": CFB_PUBLIC_FOLLOW if public else CFB_HIGH_MONEY,
            "cross": cross, "whipsaw": "partial_retrace_intact" if intact else
                "recovered" if _truthy(side.get("whipsaw_recovered")) else "none",
            "evidence": {
                "bets_pct": bets, "money_pct": money, "open_line": side["open_line"],
                "current_line": side["current_line"], "open_observed_at": str(open_at),
                "source_latest_at": str(side.get("source_latest_at", "")),
                "favorable_spread_move": opened - current, "cross_market": cross,
                "held_favorable_for_30_minutes": high,
            },
        })
    return candidates[0] if len(candidates) == 1 else None


def _nba_public_selection(game: pd.DataFrame, now: pd.Timestamp) -> tuple | None:
    # The verified replay vetoes a raw legacy Favorite before any hold switch
    # is applied. A hypothetical legacy promotion is not needed for this veto.
    if game["red_fox_favorite"].eq("true").any() or game["market_display"].eq("SPREAD").sum() != 1:
        return None
    if _nba_cross_mismatch(game):
        return None
    for index, row in game.loc[game["market_display"].eq("MONEYLINE")].iterrows():
        support = _side_identity(row.get("supported_side"))
        for side in _sides(row):
            name = side["flagged_side"]
            if support and support != _side_identity(name):
                continue
            price = _line_value(side.get("current_line"), "MONEYLINE")
            if price is None or not -165 <= price <= -100 or not _base_quality(side):
                continue
            bets, money = _number(side.get("bets_pct")), _number(side.get("money_pct"))
            if bets is None or money is None or bets < 70 or money < 55:
                continue
            spread = _matching_market_side(game, "SPREAD", name)
            if not spread or not _base_quality(spread):
                continue
            number = _line_value(spread.get("current_line"), "SPREAD")
            if number is None or not -4 <= number <= 0:
                continue
            opened = _line_value(side.get("open_line"), "MONEYLINE")
            if opened is None or opened == 0:
                continue
            movement = _american_implied_probability(price) - _american_implied_probability(opened)
            if not -2.5 < movement < 2.5:
                continue
            observed = pd.to_datetime(side.get("source_latest_at"), errors="coerce", utc=True)
            if pd.isna(observed) or observed > now:
                continue
            evidence = {key: side.get(key, "") for key in ("current_line", "open_line", "bets_pct", "money_pct", "reaction", "path")}
            evidence.update(spread=number, spread_source_observed_at=spread.get("source_latest_at", ""),
                            ml_source_observed_at=side["source_latest_at"], raw_implied_movement=movement)
            return index, name, evidence
    return None


def _nba_cross_mismatch(game: pd.DataFrame) -> bool:
    return any(game.get(field, pd.Series(dtype=str)).map(_truthy).any()
               for field in ("cross_market_mismatch", "cross_market_opener_mismatch_verified"))


def _apply_nba_release(current: pd.DataFrame, tracking: pd.DataFrame, at: str,
                       pilot_enabled: bool, hold_legacy: bool) -> pd.DataFrame:
    versions = tracking.get("favorite_rule_version", pd.Series(dtype=str))
    if current.empty or not (pilot_enabled or hold_legacy or versions.eq(NBA_PUBLIC_VERSION).any()):
        return current
    nba = current.loc[current["sport"].eq("nba")]
    if nba.empty:
        return current
    now = pd.Timestamp(at)
    latest = {}
    occupied = {}
    if not tracking.empty:
        history = tracking.loc[tracking["sport"].eq("nba") & tracking["game_id"].isin(nba["game_id"])].copy()
        stamps = pd.to_datetime(history["recorded_at"], errors="coerce", utc=True, format="mixed")
        history = history.loc[stamps.le(now)].sort_values("recorded_at", kind="mergesort")
        for _, state in history.loc[history["favorite_state"].eq("qualified")].drop_duplicates("game_id").iterrows():
            occupied[state["game_id"]] = state
        for _, state in history.drop_duplicates(["game_id", "market_display"], keep="last").iterrows():
            if state["favorite_state"] == "qualified":
                latest.setdefault(state["game_id"], []).append(state)
    for event, game in nba.groupby("game_id", sort=False):
        active = latest.get(event, [])
        first = occupied.get(event)
        first_pilot = nba_public_favorite(first)
        active_pilot = any(nba_public_favorite(state) for state in active)
        if hold_legacy or first_pilot:
            legacy_sides = {_side_identity(state["favorite_side"]) for state in active if not nba_public_favorite(state)}
            for index, row in game.loc[game["red_fox_favorite"].eq("true")].iterrows():
                if _side_identity(row.get("favorite_side")) not in legacy_sides:
                    _clear_favorite(current, index, "New legacy NBA Favorite promotion withheld.")
        if active_pilot and _nba_cross_mismatch(game):
            current.loc[game.index, "_nba_release_mismatch"] = "true"
        if not (pilot_enabled or active_pilot) or (first is not None and not first_pilot):
            continue
        pick = _nba_public_selection(game, now)
        if not pick:
            continue
        index, name, evidence = pick
        row = game.loc[index]
        if first_pilot and _side_identity(first.get("favorite_side")) != _side_identity(name):
            continue
        kickoff = pd.to_datetime(row.get("kickoff_iso"), errors="coerce", utc=True)
        if pd.isna(kickoff) or kickoff <= now or (not active_pilot and not 40 < (kickoff - now).total_seconds() / 60 <= 180):
            continue
        values = dict(red_fox_favorite="true", favorite_side=name, favorite_pathway=NBA_PUBLIC_PATH,
                      favorite_rule_version=NBA_PUBLIC_VERSION, favorite_first_qualified_at=at,
                      favorite_final_qualified_at=at, favorite_state="qualified",
                      favorite_originally_qualified="true", favorite_snapshot_id=_snapshot_id(row, at),
                      favorite_supporting_evidence=json.dumps(evidence, separators=(",", ":")),
                      favorite_final_market_read=str(row.get("reaction", "")), favorite_final_market_rank=str(row.get("board_rank", "")),
                      favorite_reason="NBA Short Public Neutral Moneyline qualified.")
        for column, value in values.items():
            current.at[index, column] = value
    return current


def _apply_cfb_spread_pathways(current: pd.DataFrame, tracking: pd.DataFrame,
                               market_history: pd.DataFrame | None, at: str,
                               enabled: frozenset[str]) -> pd.DataFrame:
    if current.empty or not {"sport", "game_id", "market_display"}.issubset(current):
        return current
    versions = tracking.get("favorite_rule_version", pd.Series(dtype=str))
    if not enabled and not versions.isin(CFB_SPREAD_VERSIONS.values()).any():
        return current
    now = pd.to_datetime(at, utc=True)
    cfb = {"ncaaf", "cfb"}
    active = {}
    first_confirmed_sides = {}
    if not tracking.empty:
        timestamps = pd.to_datetime(tracking["recorded_at"], errors="coerce", utc=True, format="mixed")
        eligible = tracking.loc[timestamps.le(now) & tracking["sport"].isin(cfb)].sort_values(
            "recorded_at", kind="mergesort"
        )
        # An additional pathway may return after a legitimate falloff, but
        # an event that already published one side cannot add its opponent.
        first = eligible.loc[eligible["favorite_state"].eq("qualified") &
                             eligible["favorite_rule_version"].isin(CFB_SPREAD_VERSIONS.values())]
        for _, state in first.drop_duplicates(["sport", "game_id"], keep="first").iterrows():
            first_confirmed_sides[(state["sport"], state["game_id"])] = _side_identity(state["favorite_side"])
        latest = eligible.drop_duplicates(["sport", "game_id", "market_display"], keep="last")
        for _, state in latest.loc[latest["favorite_state"].eq("qualified")].iterrows():
            active.setdefault((state["sport"], state["game_id"]), []).append(state)
    histories = {}
    if (market_history is not None and not market_history.empty and
            (CFB_HIGH_MONEY in enabled or versions.eq(CFB_SPREAD_VERSIONS[CFB_HIGH_MONEY]).any())):
        mask = market_history["sport"].isin(cfb) & market_history["market_display"].eq("SPREAD")
        histories = dict(tuple(market_history.loc[mask].groupby(["sport", "game_id"], sort=False)))
    for key, game in current.loc[current["sport"].isin(cfb)].groupby(["sport", "game_id"], sort=False):
        prior = active.get(key, [])
        if any(not cfb_spread_favorite(state) for state in prior) or game["red_fox_favorite"].map(_truthy).any():
            continue
        spreads = game.loc[game["market_display"].eq("SPREAD")]
        if len(spreads) != 1:
            continue
        index = spreads.index[0]
        row = spreads.iloc[0]
        if (*key, "SPREAD") in VISIBILITY_INVALIDATED_FAVORITES or _manual_favorite_exclusion(row):
            continue
        # Turning off an activation switch stops new promotions. Already active
        # selections still run their own rule through ordinary safety and lock.
        allowed = frozenset(state["favorite_pathway"] for state in prior) if prior else enabled
        decision = _cfb_spread_decision(row, game, histories.get(key, pd.DataFrame()), now, allowed,
                                        first_confirmed_sides.get(key, ""))
        if not decision:
            if prior:
                current.at[index, "_ncaaf_decision_reason"] = "The active CFB spread pathway failed its current qualification gates."
            continue
        path = decision["pathway"]
        description = "CFB Public-Backed Spread Follow" if path == CFB_PUBLIC_FOLLOW else "CFB Low-Ticket / High-Money Spread"
        values = {
            "red_fox_favorite": "true", "favorite_side": decision["side"], "favorite_pathway": path,
            "favorite_rule_version": CFB_SPREAD_VERSIONS[path], "favorite_first_qualified_at": at,
            "favorite_final_qualified_at": at, "favorite_state": "qualified",
            "favorite_final_market_read": " | ".join(filter(None, (str(row.get("reaction", "")), str(row.get("path", ""))))),
            "favorite_final_market_rank": str(row.get("board_rank", "")),
            "favorite_supporting_evidence": json.dumps(decision["evidence"], separators=(",", ":")),
            "favorite_whipsaw_state": decision["whipsaw"], "favorite_cross_market_state": decision["cross"],
            "favorite_snapshot_id": _snapshot_id(row, at), "favorite_reason": description + " qualified.",
            "favorite_originally_qualified": "true",
        }
        for column, value in values.items():
            current.at[index, column] = value
    return current


def _resolve_cfb_spread_conflicts(current: pd.DataFrame, at: str) -> None:
    experiment_versions = current.get("favorite_rule_version", pd.Series(dtype=str)).isin(CFB_SPREAD_VERSIONS.values())
    if not experiment_versions.any():
        return
    favorite = current["red_fox_favorite"].map(_truthy)
    legacy_games = set(zip(current.loc[favorite & ~experiment_versions, "sport"],
                          current.loc[favorite & ~experiment_versions, "game_id"]))
    for index, row in current.loc[favorite & experiment_versions].iterrows():
        if (row["sport"], row["game_id"]) in legacy_games:
            reason = "Existing CFB Favorite took precedence over the additional spread pathway."
            _late_invalidate_favorite(current, index, reason, at)
            _set_review(current, index, "late_invalidated", reason, at)


def _ncaaf_favorite_gate(row: pd.Series, side: dict, cross_state: str) -> dict:
    """Apply NCAAF-only market-confirmation gates without changing Market Reads."""
    market = "SPREAD"
    opened = _line_value(side.get("open_line"), market)
    current = _line_value(side.get("current_line"), market)
    move = abs(current - opened) if opened is not None and current is not None else 0.0
    context = _parts(side.get("context_chips")) | _parts(side.get("anomaly_chips"))
    crossed = set(re.findall(r"\d+(?:\.\d+)?", str(side.get("key_numbers_crossed", ""))))
    market_move = "Market Move" in context
    key_exception = (
        move >= CONFIG.ncaaf_key_number_move_min
        and bool(crossed & {"3", "7"})
        and cross_state == "confirmation"
    )
    standard_move = move >= CONFIG.ncaaf_market_move_min and market_move
    move_gate = standard_move or key_exception
    status = "confirmed" if standard_move else ("key_number_exception" if key_exception else "missing")

    if str(side.get("response_direction", "")).upper() != "TOWARD":
        return _ncaaf_gate_result(False, move_gate, status, move, "NCAAF Favorite withheld: movement is not toward the candidate side.")
    open_at = pd.to_datetime(side.get("open_observed_at", ""), errors="coerce", utc=True)
    kickoff = pd.to_datetime(row.get("kickoff_iso", ""), errors="coerce", utc=True)
    if pd.notna(open_at) and pd.notna(kickoff) and open_at < kickoff - pd.Timedelta(days=CONFIG.ncaaf_active_market_window_days):
        return _ncaaf_gate_result(
            False, move_gate, "stale_open", move,
            "NCAAF Favorite withheld: the retained opener predates the active game-week window and lacks a rebased current confirmation.",
        )
    if not move_gate:
        return _ncaaf_gate_result(
            False, False, status, move,
            f"NCAAF Favorite withheld: requires a confirmed {CONFIG.ncaaf_market_move_min:g}-point move toward the side "
            "from the active-window opener, or a confirmed 1-point move across 3 or 7.",
        )
    short_spread = current is not None and abs(current) <= 3
    if short_spread and cross_state != "confirmation":
        return _ncaaf_gate_result(
            False, True, "short_spread_missing_cross_market_confirmation", move,
            "NCAAF Favorite withheld: a spread of ±3 or less requires affirmative Moneyline/cross-market confirmation.",
        )
    recovered = _truthy(side.get("whipsaw_recovered")) or "Whipsaw Recovered" in context
    intact_partial_whipsaw = _intact_ncaaf_partial_whipsaw(side)
    if str(side.get("path", "")) == "Whipsaw" and not recovered and not intact_partial_whipsaw:
        return _ncaaf_gate_result(
            False, True, "unresolved_whipsaw", move,
            "NCAAF Favorite withheld: whipsaw movement has not retained at least 80% of the move, "
            "stayed within six material direction changes, and held for 30 minutes.",
        )
    if intact_partial_whipsaw:
        status = "stable_partial_whipsaw"
        reason = "NCAAF Favorite confirmed by a materially intact whipsaw move that held for at least 30 minutes."
    elif key_exception and not standard_move:
        reason = "NCAAF key-number exception confirmed: at least 1 point across 3 or 7 with Moneyline/cross-market support."
    else:
        reason = "NCAAF Favorite confirmed by a persistent market-supported move."
    return _ncaaf_gate_result(True, True, status, move, reason)


def _ncaaf_gate_for_market(
    row: pd.Series,
    side: dict,
    game_rows: pd.DataFrame | None,
) -> tuple[dict, str]:
    market = str(row.get("market_display", "")).upper()
    if market == "SPREAD":
        cross_state = _corresponding_moneyline_state(side, game_rows)
        return _ncaaf_favorite_gate(row, side, cross_state), cross_state
    spread_side = _matching_market_side(game_rows, "SPREAD", side.get("flagged_side"))
    ml_confirms = (
        str(side.get("response_direction", "")).upper() == "TOWARD"
        and (_number(side.get("price_move_pct")) or 0) >= CONFIG.meaningful_moneyline_price_move
    )
    cross_state = "confirmation" if ml_confirms else "neutral"
    if spread_side:
        return _ncaaf_favorite_gate(row, spread_side, cross_state), cross_state
    return _ncaaf_gate_result(
        False, False, "missing_spread", 0.0,
        "NCAAF Favorite withheld: Moneyline expression lacks a qualifying corresponding spread move.",
    ), cross_state


def _ncaaf_gate_result(passed: bool, move_gate: bool, status: str, move: float, reason: str) -> dict:
    return {
        "passed": passed,
        "move_gate_passed": move_gate,
        "market_move_status": status,
        "move": round(move, 3),
        "reason": reason,
    }


def _pathway(side: dict, other: dict, sport: str, market: str) -> str:
    bets, money = _number(side.get("bets_pct")), _number(side.get("money_pct"))
    other_bets, other_money = _number(other.get("bets_pct")), _number(other.get("money_pct"))
    if None in {bets, money, other_bets, other_money}:
        return ""
    extended_nfl_dog = (
        sport == "nfl"
        and market == "SPREAD"
        and (_line_value(side.get("current_line"), market) or 0) > CONFIG.spread_max
        and bets <= CONFIG.nfl_extended_spread_max_bets
        and money <= CONFIG.nfl_extended_spread_max_money
    )
    low = (
        (bets <= CONFIG.low_support_max and money <= CONFIG.low_support_max)
        or extended_nfl_dog
    ) and bets < other_bets and money < other_money
    toward = str(side.get("response_direction", "")).upper() == "TOWARD"
    reaction = str(side.get("reaction", ""))
    if low and reaction == "Contrarian" and toward and _truthy(side.get("kpi_eligible")):
        return "low_support_contrarian"
    sharp_money_continuation = (
        sport in {"ncaaf", "cfb"}
        and bets <= CONFIG.ncaaf_sharp_money_max_bets
        and CONFIG.ncaaf_sharp_money_min <= money <= CONFIG.ncaaf_sharp_money_max
        and money - bets >= CONFIG.ncaaf_sharp_money_min_gap
        and bets < other_bets
        and money > other_money
    )
    if (
        sharp_money_continuation
        and reaction == "Contrarian"
        and toward
        and _truthy(side.get("kpi_eligible"))
    ):
        return "low_ticket_high_money_contrarian"
    if low and _is_favorite_resistance(side, other, sport, market):
        return "low_support_freeze"
    moderate = (
        CONFIG.moderate_support_min <= bets <= CONFIG.moderate_support_max
        and CONFIG.moderate_support_min <= money <= CONFIG.moderate_support_max
    )
    if moderate and toward and _has_meaningful_move(side):
        return "moderate_support_follow"
    return ""


def _shadow_freeze_candidate(row: pd.Series, game_rows: pd.DataFrame | None) -> dict | None:
    """Identify the approved NFL 75-79% pressure cohort without publishing a Favorite."""
    if str(row.get("sport", "")).lower() != "nfl" or str(row.get("market_display", "")).upper() != "SPREAD":
        return None
    sides = _sides(row)
    if len(sides) != 2:
        return None
    for pressure in sides:
        bets = _number(pressure.get("bets_pct"))
        money = _number(pressure.get("money_pct"))
        if (
            bets is None or money is None or not 75 <= bets < 80 or money < 80
            or str(pressure.get("reaction", "")) != "Freeze"
            or str(pressure.get("path", "")) != "Held"
            or not str(pressure.get("key_number_pinned", "")).strip()
            or not _base_quality(pressure)
        ):
            continue
        candidate = next((side for side in sides if side is not pressure), None)
        if candidate is None or _corresponding_moneyline_state(candidate, game_rows) == "contradiction":
            continue
        return {
            "side": str(candidate.get("flagged_side", "")),
            "reason": (
                "Shadow only: 75-79% NFL bet pressure with at least 80% money held at a key number "
                "without a Moneyline contradiction."
            ),
        }
    return None


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
    return not _quality_failure(side)


def _quality_failure(side: dict) -> str:
    if str(side.get("data_badge", "")) != "Clean":
        return "Market data was not Clean."
    if (_number(side.get("observation_count")) or 0) < CONFIG.minimum_observations:
        return f"Market had fewer than {CONFIG.minimum_observations} observations."
    context = _parts(side.get("context_chips"))
    risks = sorted({"Market Lag", "Feed Risk", "Split Risk", "Split Cap"} & context)
    if risks:
        return f"Blocking market context: {', '.join(risks)}."
    if _truthy(side.get("active_worsening_reversal")):
        return "Market showed an active worsening reversal."
    if str(side.get("path", "")) == "Whipsaw" and not _truthy(side.get("whipsaw_recovered")):
        if not _intact_partial_whipsaw(side):
            return "Unrecovered Whipsaw failed the intact-move exception."
    elif _truthy(side.get("return_toward_open")):
        return "Movement retraced toward the opener."
    return ""


def _intact_partial_whipsaw(side: dict) -> bool:
    """Allow a retrace only when most of a meaningful move remains and has held."""
    if str(side.get("path", "")) != "Whipsaw" or _truthy(side.get("whipsaw_recovered")):
        return False
    retention = _number(side.get("whipsaw_retention_ratio"))
    if retention is None:
        current_move = _number(side.get("move_abs"))
        maximum = _number(side.get("max_excursion"))
        retention = current_move / maximum if current_move is not None and maximum and maximum > 0 else None
    material_changes = _number(side.get("material_direction_changes"))
    if material_changes is None:
        material_changes = _number(side.get("line_dir_changes"))
    return bool(
        retention is not None
        and retention >= CONFIG.partial_whipsaw_min_retention
        and material_changes is not None
        and material_changes <= CONFIG.partial_whipsaw_max_material_direction_changes
        and str(side.get("response_direction", "")).upper() == "TOWARD"
        and _has_meaningful_move(side)
        and _truthy(side.get("whipsaw_confirmation_ready"))
        and (_number(side.get("whipsaw_confirmation_minutes")) or 0) >= CONFIG.partial_whipsaw_confirmation_minutes
    )


def _intact_ncaaf_partial_whipsaw(side: dict) -> bool:
    """Allow a mature CFB move without requiring the path label to fully recover.

    The ordinary partial-whipsaw gate remains unchanged for every other sport.
    CFB receives a slightly wider churn allowance only when at least 80% of a
    meaningful move remains, the move is still toward the candidate, and the
    final state has held for 30 minutes. A return toward the opener remains a
    hard rejection.
    """
    if str(side.get("path", "")) != "Whipsaw" or _truthy(side.get("whipsaw_recovered")):
        return False
    retention = _number(side.get("whipsaw_retention_ratio"))
    material_changes = _number(side.get("material_direction_changes"))
    if material_changes is None:
        material_changes = _number(side.get("line_dir_changes"))
    return bool(
        retention is not None
        and retention >= CONFIG.ncaaf_partial_whipsaw_min_retention
        and material_changes is not None
        and material_changes <= CONFIG.ncaaf_partial_whipsaw_max_material_direction_changes
        and not _truthy(side.get("return_toward_open"))
        and not _truthy(side.get("active_worsening_reversal"))
        and str(side.get("response_direction", "")).upper() == "TOWARD"
        and _has_meaningful_move(side)
        and _truthy(side.get("whipsaw_confirmation_ready"))
        and (_number(side.get("whipsaw_confirmation_minutes")) or 0)
        >= CONFIG.ncaaf_partial_whipsaw_confirmation_minutes
    )


def _number_is_eligible(sport: str, market: str, value: float, side: dict, game_rows: pd.DataFrame | None) -> bool:
    if market == "SPREAD":
        if CONFIG.spread_min <= abs(value) <= CONFIG.spread_max:
            return True
        bets = _number(side.get("bets_pct"))
        money = _number(side.get("money_pct"))
        crossed = _parts(side.get("key_numbers_crossed"))
        extended_nfl_dog = (
            sport == "nfl"
            and CONFIG.spread_max < value <= CONFIG.nfl_positive_spread_max
            and bets is not None and money is not None
            and bets <= CONFIG.nfl_extended_spread_max_bets
            and money <= CONFIG.nfl_extended_spread_max_money
            and (
                (_number(side.get("line_move_abs")) or 0) >= CONFIG.nfl_extended_spread_min_move
                or "K10" in crossed
            )
            and not _truthy(side.get("active_worsening_reversal"))
        )
        return extended_nfl_dog
    maximum = CONFIG.primary_moneyline_max if sport in CONFIG.primary_moneyline_sports else CONFIG.moneyline_max
    if not CONFIG.moneyline_min <= value <= maximum:
        return False
    if sport == "mlb" and not _mlb_plus_money_conviction_gate(side, value):
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
    if sport in {"nfl", "ncaaf", "cfb"}:
        ncaaf_sharp_money = (
            sport in {"ncaaf", "cfb"}
            and _ncaaf_sharp_money_profile(side)
            and _ncaaf_sharp_money_profile(spread)
        )
        if not ncaaf_sharp_money and not _football_favorite_moneyline_gate(
            side, spread, spread_value,
        ):
            return False
    return True


def _ncaaf_sharp_money_profile(side: dict | None) -> bool:
    if not side:
        return False
    bets = _number(side.get("bets_pct"))
    money = _number(side.get("money_pct"))
    return bool(
        bets is not None
        and money is not None
        and bets <= CONFIG.ncaaf_sharp_money_max_bets
        and CONFIG.ncaaf_sharp_money_min <= money <= CONFIG.ncaaf_sharp_money_max
        and money - bets >= CONFIG.ncaaf_sharp_money_min_gap
    )


def _mlb_plus_money_conviction_gate(side: dict, current_value: float) -> bool:
    """Require real conviction from an MLB side that opened plus money.

    An intact one-way move through pick'em can continue to the ordinary gates,
    but a partial-retrace whipsaw cannot become a Favorite after crossing from
    plus to minus money. If both opener and current price remain positive,
    require at least five implied-probability points of improvement. Negative
    openers are unaffected.
    """
    open_value = _line_value(side.get("open_line"), "MONEYLINE")
    if open_value is None or open_value <= 0:
        return True
    if current_value < 0:
        partial_retrace = (
            str(side.get("path", "")) == "Whipsaw"
            and not _truthy(side.get("whipsaw_recovered"))
            and "Whipsaw Recovered" not in _parts(side.get("context_chips"))
        )
        return not partial_retrace
    if current_value == 0:
        return True
    return (_number(side.get("price_move_pct")) or 0) >= CONFIG.mlb_still_plus_min_probability_move


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
    if _truthy(side.get("return_toward_open")) and not _intact_partial_whipsaw(side):
        return False
    if _truthy(spread.get("return_toward_open")) and not _intact_partial_whipsaw(spread):
        return False
    side_changes = (
        int(_number(side.get("material_direction_changes")) or 0)
        if _intact_partial_whipsaw(side)
        else int(_number(side.get("line_dir_changes")) or 0)
    )
    spread_changes = (
        int(_number(spread.get("material_direction_changes")) or 0)
        if _intact_partial_whipsaw(spread)
        else int(_number(spread.get("line_dir_changes")) or 0)
    )
    direction_changes = max(side_changes, spread_changes)
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


def _prefer_confirmed_football_moneyline(frame: pd.DataFrame) -> pd.DataFrame:
    """Express one short favorite thesis through its safer win condition.

    A qualifying football team laying no more than four points can use its
    negative moneyline only when that market independently qualifies, remains
    inside the approved price band, and confirms the spread move.  The spread
    is then suppressed so one correlated thesis cannot create two Favorites.
    """
    result = frame.copy()
    football = {"nfl", "ncaaf", "cfb"}
    for (sport, _game_id), game in result.groupby(["sport", "game_id"], dropna=False, sort=False):
        if str(sport).lower() not in football:
            continue
        spread_rows = game[
            game["market_display"].astype(str).str.upper().eq("SPREAD")
            & game["red_fox_favorite"].map(_truthy)
            & game["favorite_cross_market_state"].astype(str).eq("confirmation")
        ]
        moneyline_rows = game[
            game["market_display"].astype(str).str.upper().eq("MONEYLINE")
            & game["red_fox_favorite"].map(_truthy)
        ]
        for spread_index, spread_row in spread_rows.iterrows():
            identity = _side_identity(spread_row.get("favorite_side"))
            spread_side = next(
                (item for item in _sides(spread_row) if _side_identity(item.get("flagged_side")) == identity),
                None,
            )
            spread_value = _line_value(spread_side.get("current_line"), "SPREAD") if spread_side else None
            if spread_value is None or not -CONFIG.secondary_moneyline_spread_max <= spread_value < 0:
                continue
            for moneyline_index, moneyline_row in moneyline_rows.iterrows():
                if _side_identity(moneyline_row.get("favorite_side")) != identity:
                    continue
                moneyline_side = next(
                    (item for item in _sides(moneyline_row) if _side_identity(item.get("flagged_side")) == identity),
                    None,
                )
                price = _line_value(moneyline_side.get("current_line"), "MONEYLINE") if moneyline_side else None
                if price is None or not CONFIG.moneyline_min <= price < 0:
                    continue
                _clear_favorite(
                    result,
                    spread_index,
                    "Favorite expressed on the confirmed moneyline to use the safer win condition and avoid a duplicate correlated spread wager.",
                )
                result.at[moneyline_index, "favorite_cross_market_state"] = "confirmation"
                result.at[moneyline_index, "favorite_reason"] = (
                    f"Selected over the matching {spread_value:g} spread: the confirming moneyline at "
                    f"{price:+g} is inside the approved price band and has the safer win condition."
                )
                try:
                    evidence = json.loads(str(result.at[moneyline_index, "favorite_supporting_evidence"] or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    evidence = {}
                evidence["cross_market"] = "confirmation"
                evidence["preferred_over_spread"] = str(spread_row.get("favorite_side", ""))
                result.at[moneyline_index, "favorite_supporting_evidence"] = json.dumps(evidence, separators=(",", ":"))
                break
    return result


def _prefer_basketball_moneyline_expression(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate one qualifying laying-points thesis to one moneyline result.

    Unlike football, the basketball spread supplies the qualifying pathway and
    the matching moneyline supplies reliable cross-market alignment plus the
    safer expression. Requiring the moneyline to clear the complete Favorite
    pathway again would create an unnecessary second threshold and delay the
    signal until much of its price value had disappeared.
    """
    result = frame.copy()
    basketball = {"nba", "ncaab", "cbb"}
    for (sport, _game_id), game in result.groupby(["sport", "game_id"], dropna=False, sort=False):
        if str(sport).lower() not in basketball:
            continue
        spread_rows = game[
            game["market_display"].astype(str).str.upper().eq("SPREAD")
            & game["red_fox_favorite"].map(_truthy)
        ]
        moneyline_rows = game[game["market_display"].astype(str).str.upper().eq("MONEYLINE")]
        for spread_index, spread_row in spread_rows.iterrows():
            identity = _side_identity(spread_row.get("favorite_side"))
            spread_side = next(
                (item for item in _sides(spread_row) if _side_identity(item.get("flagged_side")) == identity),
                None,
            )
            spread_value = _line_value(spread_side.get("current_line"), "SPREAD") if spread_side else None
            if spread_value is None or not -CONFIG.secondary_moneyline_spread_max <= spread_value <= 0:
                continue
            moneyline_match = None
            for moneyline_index, moneyline_row in moneyline_rows.iterrows():
                moneyline_side = next(
                    (item for item in _sides(moneyline_row) if _side_identity(item.get("flagged_side")) == identity),
                    None,
                )
                price = _line_value(moneyline_side.get("current_line"), "MONEYLINE") if moneyline_side else None
                if (
                    moneyline_side is not None
                    and price is not None
                    and CONFIG.moneyline_min <= price < 0
                    and _side_identity(moneyline_row.get("supported_side")) == identity
                    and _basketball_moneyline_alignment(moneyline_side) == "confirmation"
                ):
                    moneyline_match = (moneyline_index, moneyline_row, moneyline_side, price)
                    break
            if moneyline_match is None:
                _clear_favorite(
                    result,
                    spread_index,
                    "Basketball Favorite withheld: the supported team is laying points but no eligible, aligned moneyline expression is available.",
                )
                continue
            moneyline_index, moneyline_row, moneyline_side, price = moneyline_match
            spread_values = {column: result.at[spread_index, column] for column in FAVORITE_COLUMNS}
            _clear_favorite(
                result,
                spread_index,
                "Favorite expressed on the confirmed basketball moneyline to avoid a duplicate correlated spread wager.",
            )
            if not _truthy(result.at[moneyline_index, "red_fox_favorite"]):
                for column, value in spread_values.items():
                    result.at[moneyline_index, column] = value
            result.at[moneyline_index, "red_fox_favorite"] = "true"
            result.at[moneyline_index, "favorite_side"] = str(moneyline_side.get("flagged_side", ""))
            result.at[moneyline_index, "favorite_rule_version"] = CONFIG.basketball_version
            result.at[moneyline_index, "favorite_state"] = "qualified"
            result.at[moneyline_index, "favorite_cross_market_state"] = "confirmation"
            result.at[moneyline_index, "favorite_final_market_rank"] = str(moneyline_row.get("board_rank", ""))
            result.at[moneyline_index, "favorite_snapshot_id"] = _snapshot_id(
                moneyline_row,
                str(result.at[moneyline_index, "favorite_final_qualified_at"]),
            )
            result.at[moneyline_index, "favorite_reason"] = (
                f"Selected over the matching {spread_value:g} basketball spread: the 1.5-point-or-greater "
                f"spread signal is confirmed by aligned moneyline support; the team is laying points, "
                f"so {price:+g} is the sole Favorite expression inside the approved price band."
            )
            try:
                evidence = json.loads(str(result.at[moneyline_index, "favorite_supporting_evidence"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                evidence = {}
            evidence.update({
                "cross_market": "confirmation",
                "basketball_expression": "moneyline",
                "expression_price": price,
                "preferred_over_spread": str(spread_row.get("favorite_side", "")),
            })
            result.at[moneyline_index, "favorite_supporting_evidence"] = json.dumps(
                evidence, separators=(",", ":"),
            )
    return result


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


def _matching_market_row(
    game_rows: pd.DataFrame | None,
    market: str,
    side_name: object,
) -> pd.Series | None:
    if game_rows is None or game_rows.empty:
        return None
    target = _side_identity(side_name)
    rows = game_rows[game_rows["market_display"].astype(str).str.upper().eq(market)]
    for _, row in rows.iterrows():
        if any(_side_identity(side.get("flagged_side")) == target for side in _sides(row)):
            return row
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
            if _truthy(row.get("favorite_late_invalidated")) and mask.any():
                for column in (
                    "favorite_originally_qualified", "favorite_late_invalidated",
                    "favorite_late_invalidated_at", "favorite_late_invalidated_reason",
                ):
                    updated.loc[mask, column] = row.get(column, "")
            if _truthy(row.get("_visibility_restored")):
                continue
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
        "favorite_pathway": row.get("favorite_pathway", ""),
        "favorite_rule_version": row.get(
            "favorite_rule_version", _favorite_rule_version(row.get("sport", ""))
        ),
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
        "source_observed_at": _review_source_observed_at(row, row.get("favorite_side", "")),
        "transition_reason": str(row.get("favorite_review_reason") or row.get("favorite_reason") or ""),
    }
