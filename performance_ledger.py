"""Build an admin-only performance ledger from already-frozen market records.

This module is deliberately outside the snapshot, scoring, ranking, and board
publication path.  It consumes Live & Recent's first-write-wins pregame rows,
then attaches scores already collected by report maintenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests
from mlb_fixtures import SCORE_SOURCE as MLB_SCORE_SOURCE, cutover_at as mlb_cutover_at, timestamp as mlb_timestamp
from manual_favorites import HANDOFF as MANUAL_HANDOFF, VERSION as MANUAL_VERSION

from anomaly_action_results import _grade_action
from nhl_release_config import OFFICIAL_HANDOFF, OFFICIAL_VERSIONS, official_nhl_favorite
import ufc_bout_integrity as ufc
from red_fox_favorite import (CFB_SPREAD_VERSIONS, NBA_PUBLIC_VERSION, MLB_STATIC_VERSION, NOV24_VERSION,
                              cfb_spread_favorite, nba_public_favorite, mlb_static_favorite, tracked_entry_quote)


SUPPORTED_SIDE_VALID_FROM = pd.Timestamp("2026-09-07T17:57:58Z")
FAVORITE_VALID_FROM = pd.Timestamp("2026-09-07T22:11:12Z")
FAVORITE_TRACKING_START_DATE = "2026-09-07"
# Initial production cutovers for the dated, sport-specific Favorite releases.
NOV24_LOGIC_ACTIVATED_AT = pd.Timestamp("2026-11-24T22:39:02Z")  # 88fa199
FAVORITE_LOGIC_CUTOVERS = {
    "ncaab": NOV24_LOGIC_ACTIVATED_AT,
    "nba": NOV24_LOGIC_ACTIVATED_AT,
    "nhl": NOV24_LOGIC_ACTIVATED_AT,
    "ncaaf": pd.Timestamp("2026-11-25T05:19:35Z"),  # aab6888, CFB spread release
    "mlb": pd.Timestamp("2026-11-25T18:13:33Z"),  # a13bcf0, automatic Static ML rollout
    "ufc": pd.Timestamp("2026-11-25T18:40:19Z"),  # verified-bout production activation
}
FAVORITE_SPORTS = (
    ("mlb", "MLB"),
    ("nba", "NBA"),
    ("ncaab", "NCAAB"),
    ("ncaaf", "CFB"),
    ("nfl", "NFL"),
    ("nhl", "NHL"),
    ("ufc", "UFC"),
)
FAVORITE_RULE_VERSIONS = frozenset({
    "red_fox_favorite_v2",
    "red_fox_favorite_v3",
    "red_fox_favorite_v4",
    "red_fox_favorite_v5",
    "red_fox_favorite_v6",
    "red_fox_favorite_v7",
    "red_fox_favorite_v8",
    "red_fox_favorite_basketball_v1",
    "red_fox_favorite_basketball_v2",
    "red_fox_favorite_ufc_v1",
    "red_fox_favorite_nhl_v1",
    NOV24_VERSION,
    *CFB_SPREAD_VERSIONS.values(),
    MLB_STATIC_VERSION,
    NBA_PUBLIC_VERSION,
    MANUAL_VERSION,
    *OFFICIAL_VERSIONS.values(),
})
FAVORITE_KPI_EXCLUSIONS = {
    ("34877187", "SPREAD"): (
        "Excluded as a duplicate correlated basketball thesis: Orlando was laying points, "
        "so its official Red Fox Favorite result is recorded once through the moneyline."
    ),
    ("34603681", "SPREAD"): (
        "Excluded: audited final-hour visibility failure invalidated the Favorite."
    ),
    ("34694536", "MONEYLINE"): (
        "Rule invalidated: Washington opened +123 and remained plus money at +108; "
        "the 3.2-point implied-probability move did not meet the MLB five-point standard."
    ),
    ("34726372", "MONEYLINE"): (
        "Rule invalidated after review: Miami opened +109 and crossed to negative money; "
        "the MLB plus-to-minus Favorite cohort is suspended from official Favorites."
    ),
}
RETROACTIVE_FAVORITE_INCLUSIONS = {
    ("34827209", "MONEYLINE"): {
        "team": "South Carolina",
        "first_qualified_at": "2026-10-20T03:50:09.703470+00:00",
        "final_qualified_at": "2026-10-24T19:25:59.443821+00:00",
        "corrected_at": "2026-10-24T23:18:37.807937+00:00",
        "reason": (
            "Retroactive system correction: South Carolina retained a six-point spread move and "
            "confirmed plus-to-minus moneyline move while low ticket share developed concentrated "
            "money support; the former 45% money cutoff incorrectly removed the Favorite."
        ),
    },
}
LEDGER_COLUMNS = [
    "ledger_id", "event_id", "game", "sport", "scheduled_start", "market", "side",
    "open_line", "open_price", "decision_line", "decision_price", "decision_line_source",
    "decision_quote_observed_at",
    "final_pregame_line", "final_pregame_price",
    "line_move_from_open", "market_read", "market_read_detail", "market_path",
    "bets_pct", "money_pct", "bets_pct_band", "money_pct_band",
    "movement_magnitude", "movement_magnitude_band", "favorite_cross_market_state",
    "observation_count", "line_direction_changes", "return_toward_open",
    "active_worsening_reversal", "whipsaw_state", "whipsaw_ever",
    "supported_side", "favorite_qualified", "favorite_rule_version", "favorite_origin",
    "manual_revision", "manual_provider_event_id",
    "favorite_originally_qualified", "favorite_late_invalidated",
    "favorite_late_invalidated_at", "favorite_late_invalidated_reason",
    "favorite_pathway", "candidate_decision", "candidate_decision_reason",
    "price_band", "spread_band",
    "favorite_first_qualified_at", "favorite_final_qualified_at",
    "favorite_first_fell_off_at", "favorite_last_fell_off_at",
    "favorite_fell_off_before_kickoff", "favorite_ever_fell_off", "favorite_final_state",
    "favorite_t20_state", "favorite_locked_decision", "qualification_episode_count",
    "requalification_count", "falloff_count", "qualifying_capture_count",
    "minutes_first_qualified_before_start", "minutes_final_state_before_start",
    "first_qualified_timing_band", "final_state_timing_band",
    "final_pregame_state_at_utc", "final_pregame_frozen_at",
    "classification_correction", "classification_corrected_at_utc",
    "classification_correction_reason", "classification_original_publication",
    "final_score", "market_result", "grade", "stake_units", "net_profit_units", "roi",
    "closing_line", "closing_price", "closing_line_source", "closing_price_source",
    "closing_line_observed_at", "closing_price_observed_at",
    "clv", "clv_method", "clv_available",
    "score_provider", "score_source", "score_provider_event_id", "score_resolved_at",
    "graded_at_utc", "grade_evidence_hash",
    "freeze_source", "freeze_method", "source_snapshot_id", "source_hash", "audit_status",
]
CLASSIFICATION_COLUMNS = LEDGER_COLUMNS[:LEDGER_COLUMNS.index("final_score")] + [
    "freeze_source", "freeze_method", "source_snapshot_id", "source_hash",
]
RESULT_COLUMNS = [
    "final_score", "market_result", "grade", "stake_units", "net_profit_units", "roi",
    "closing_line", "closing_price", "closing_line_source", "closing_price_source",
    "closing_line_observed_at", "closing_price_observed_at",
    "clv", "clv_method", "clv_available",
    "score_provider", "score_source", "score_provider_event_id", "score_resolved_at",
    "graded_at_utc", "grade_evidence_hash", "audit_status",
]
EXPORT_FILES = {
    "combined": "performance_combined.csv",
    "supported": "performance_supported_sides.csv",
    "favorites": "performance_favorites.csv",
    "candidate_audit": "favorite_candidate_audit.csv",
    "cohort_kpis": "favorite_cohort_kpis.csv",
}

COHORT_DIMENSIONS = [
    "sport", "market", "favorite_pathway", "market_read", "bets_pct_band", "money_pct_band",
    "price_band", "spread_band", "movement_magnitude_band", "favorite_cross_market_state",
    "first_qualified_timing_band", "final_state_timing_band", "whipsaw_ever",
    "favorite_ever_fell_off", "favorite_t20_state", "favorite_locked_decision",
    "favorite_rule_version",
]
COHORT_KPI_COLUMNS = [
    "dimension", "segment", "candidate_decision", "candidates", "graded",
    "wins", "losses", "pushes", "win_rate_excluding_pushes", "stake_units",
    "net_profit_units", "roi", "clv_available", "clv_coverage", "average_clv",
    "whipsaw_candidates", "whipsaw_rate", "fell_off_candidates", "falloff_rate",
    "sample_status",
]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() and path.stat().st_size else pd.DataFrame()
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _truthy(value: object) -> bool:
    return _text(value).lower() in {"1", "true", "yes"}


def _utc_series(values: object, index: pd.Index) -> pd.Series:
    """Return an explicitly UTC-aware Series, including all-NaT inputs."""
    source = values if isinstance(values, pd.Series) else pd.Series(values, index=index)
    parsed = pd.Series(pd.to_datetime(source, errors="coerce", utc=True), index=index)
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize("UTC")
    # pandas 3 preserves second-resolution datetimes for all-missing inputs.
    # Keep the ledger contract stable at nanosecond UTC regardless of input.
    return parsed.astype("datetime64[ns, UTC]")


def _identity(value: object) -> str:
    text = re.sub(r"\s[+-]\d+(?:\.\d+)?(?:\s.*)?$", "", _text(value))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _side_record(row: pd.Series, supported_side: str) -> dict:
    try:
        sides = json.loads(_text(row.get("market_sides")) or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        sides = []
    target = _identity(supported_side)
    matches = [side for side in sides if isinstance(side, dict) and _identity(side.get("flagged_side")) == target]
    return matches[0] if len(matches) == 1 else {}


def _line_and_price(market: str, side: str, display: object) -> tuple[str, str]:
    value = _text(display)
    price_match = re.search(r"\(([+-]?\d{3,4})\)\s*$", value)
    if market == "MONEYLINE":
        moneyline = re.search(r"(?<!\d)([+-]\d{3,4})(?!\d)", value)
        return "", moneyline.group(1) if moneyline else (price_match.group(1) if price_match else "")
    number = re.search(r"(?:^|\s)([+-]?\d+(?:\.\d+)?)", value)
    if not number:
        number = re.search(r"(?:over|under|o|u)\s*([+-]?\d+(?:\.\d+)?)", side, re.I)
    return (number.group(1) if number else "", price_match.group(1) if price_match else "")


def _side_at_decision_line(side: object, market: str, decision_line: object) -> str:
    """Return the immutable grading side at first confirmed qualification.

    ``supported_side`` describes the final pregame market state and may move.
    ``side`` is the official performance record, so spread grading must use the
    line captured when the Favorite first became confirmed.
    """
    value = _text(side)
    if _text(market).upper() != "SPREAD" or not _text(decision_line):
        return value
    team = re.sub(r"\s[+-]\d+(?:\.\d+)?(?:\s.*)?$", "", value).strip()
    return f"{team} {_text(decision_line)}" if team else value


def _number(value: object) -> float | None:
    match = re.search(r"[+-]?\d+(?:\.\d+)?", _text(value).replace(",", ""))
    try:
        return float(match.group()) if match else None
    except ValueError:
        return None


def _minutes_before(start: object, observed: object) -> str:
    kickoff = pd.to_datetime(start, errors="coerce", utc=True)
    at = pd.to_datetime(observed, errors="coerce", utc=True)
    if pd.isna(kickoff) or pd.isna(at):
        return ""
    return str(round((kickoff - at).total_seconds() / 60.0, 3))


def _timing_band(minutes: object) -> str:
    value = _number(minutes)
    if value is None:
        return "unavailable"
    if value < 0:
        return "post_start_invalid"
    if value <= 20:
        return "T-0_to_20"
    if value <= 40:
        return "T-20_to_40"
    if value <= 90:
        return "T-40_to_90"
    if value <= 240:
        return "T-90_to_240"
    return "earlier_than_T-240"


def _price_band(price: object) -> str:
    value = _number(price)
    if value is None:
        return "unavailable"
    if value < -165:
        return "below_-165"
    if value <= -151:
        return "-165_to_-151"
    if value <= -111:
        return "-150_to_-111"
    if value <= 100:
        return "-110_to_+100"
    if value <= 120:
        return "+101_to_+120"
    if value <= 125:
        return "+121_to_+125"
    return "above_+125"


def _spread_band(line: object, market: str) -> str:
    if market != "SPREAD":
        return "not_applicable"
    value = _number(line)
    if value is None:
        return "unavailable"
    magnitude = abs(value)
    if magnitude == 0:
        return "pickem"
    if magnitude <= 2.5:
        return "0.5_to_2.5"
    if magnitude <= 4.5:
        return "3_to_4.5"
    if magnitude <= 6.5:
        return "5_to_6.5"
    if magnitude <= 8:
        return "7_to_8"
    return "above_8"


def _percent_band(value: object) -> str:
    number = _number(value)
    if number is None:
        return "unavailable"
    if number <= 20:
        return "0_to_20"
    if number <= 35:
        return "21_to_35"
    if number <= 45:
        return "36_to_45"
    if number <= 55:
        return "46_to_55"
    if number <= 70:
        return "56_to_70"
    if number <= 85:
        return "71_to_85"
    return "86_to_100"


def _movement_band(value: object) -> str:
    number = _number(value)
    if number is None:
        return "unavailable"
    magnitude = abs(number)
    if magnitude == 0:
        return "held"
    if magnitude < 0.5:
        return "under_0.5"
    if magnitude < 1:
        return "0.5_to_under_1"
    if magnitude < 2:
        return "1_to_under_2"
    if magnitude < 5:
        return "2_to_under_5"
    return "5_plus"


def _candidate_pathway(side: dict, market_sides: list[dict]) -> str:
    bets, money = _number(side.get("bets_pct")), _number(side.get("money_pct"))
    reaction = _text(side.get("reaction"))
    direction = _text(side.get("response_direction")).upper()
    if bets is not None and money is not None and bets <= 45 and money <= 45:
        if reaction == "Contrarian" and direction == "TOWARD":
            return "low_support_contrarian"
        pressure = next((item for item in market_sides if item is not side), {})
        if _text(pressure.get("reaction")) == "Freeze":
            return "low_support_freeze"
    if (
        bets is not None and money is not None
        and 46 <= bets <= 60 and 46 <= money <= 60 and direction == "TOWARD"
    ):
        return "moderate_support_follow"
    return "no_qualifying_pathway"


def _candidate_reason(row: pd.Series, side: dict, pathway: str, favorite: bool) -> str:
    if favorite:
        return "Accepted by final pregame Favorite rules."
    explicit = _text(row.get("favorite_reason"))
    if explicit:
        return explicit
    market = _text(row.get("market_display")).upper()
    sport = _text(row.get("sport")).lower()
    if market == "TOTAL":
        return "Rejected: totals are outside the Favorite program."
    if _text(side.get("data_badge")) != "Clean":
        return "Rejected: source data was not Clean."
    observations = int(_number(side.get("observation_count")) or 0)
    if observations < 3:
        return "Rejected: fewer than three observations."
    context = {_text(part) for part in _text(side.get("context_chips")).split("|")}
    risks = sorted({"Market Lag", "Feed Risk", "Split Risk", "Split Cap"} & context)
    if risks:
        return "Rejected: " + ", ".join(risks) + "."
    if _truthy(side.get("active_worsening_reversal")):
        return "Rejected: active worsening reversal."
    line, price = _line_and_price(market, _text(side.get("flagged_side")), side.get("current_line"))
    value = _number(price if market == "MONEYLINE" else line)
    if market == "SPREAD" and (value is None or not 1 <= abs(value) <= 8):
        return "Rejected: spread missed the base 1-to-8 band or the strict NFL +8.5-to-+10 exception."
    if market == "MONEYLINE":
        maximum = 120 if sport in {"mlb", "nhl", "ufc"} else 125
        if value is None or not -165 <= value <= maximum:
            return f"Rejected: price was outside the -165-to-{maximum:+d} band."
    if pathway == "no_qualifying_pathway":
        return "Rejected: split, Market Read, and movement did not form a qualifying pathway."
    if _truthy(row.get("cross_market_mismatch")) or _truthy(row.get("cross_market_opener_mismatch_verified")):
        return "Rejected: confirmed cross-market contradiction."
    return "Rejected by a final Favorite gate; inspect the retained evidence fields."


def _tracking_lifecycle(history: pd.DataFrame, row: pd.Series, kickoff: pd.Timestamp,
                        *, recover_entry: bool = False) -> dict[str, str]:
    empty = {
        "first_qualified": "", "last_qualified": "", "first_fell_off": "",
        "last_fell_off": "", "episodes": "0", "requalifications": "0",
        "falloffs": "0", "qualifying_captures": "0", "whipsaw_ever": "no",
        "pathway": "", "first_qualified_line": "", "entry_observed_at": "",
        "history_available": "no", "final_state": "unknown", "t20_state": "unknown",
    }
    if history.empty:
        return empty
    mask = pd.Series(True, index=history.index)
    for column in ("sport", "game_id", "market_display"):
        if column not in history:
            return empty
        mask &= history[column].astype(str).eq(_text(row.get(column)))
    scoped = history.loc[mask].copy()
    if scoped.empty:
        return empty
    if row.get("freeze_method") == ufc.HANDOFF:
        bout_id = ufc.row_proof(row).get("bout_id")
        scoped = scoped.loc[scoped.apply(lambda event: ufc.row_proof(event, "supporting_evidence").get("bout_id") == bout_id, axis=1)].copy()
    scoped["_at"] = pd.to_datetime(scoped.get("recorded_at", ""), errors="coerce", utc=True)
    scoped = scoped[scoped["_at"].notna()]
    if pd.notna(kickoff):
        scoped = scoped[scoped["_at"] <= kickoff]
    # Legacy deployments occasionally appended the exact same capture twice.
    # Preserve the raw audit file, but do not let those duplicates inflate
    # lifecycle observations or qualification counts.
    scoped = scoped.sort_values("_at", kind="mergesort").drop_duplicates(keep="last")
    if scoped.empty:
        return empty
    states = scoped.get("favorite_state", pd.Series("", index=scoped.index)).astype(str)
    qualified = states.eq("qualified")
    previous = states.shift(fill_value="")
    episode_starts = qualified & previous.ne("qualified")
    falloffs = states.eq("not_qualified") & previous.eq("qualified")
    whipsaw = (
        scoped.get("whipsaw_state", pd.Series("", index=scoped.index)).astype(str).str.strip().ne("")
        & ~scoped.get("whipsaw_state", pd.Series("", index=scoped.index)).astype(str).str.lower().eq("none")
    ) | scoped.get("path", pd.Series("", index=scoped.index)).astype(str).str.contains("whipsaw", case=False, na=False)
    pathways = scoped.loc[qualified, "favorite_pathway"] if "favorite_pathway" in scoped else pd.Series(dtype=str)
    entry_rows = scoped.loc[qualified]
    if {"entry_observed_at", "favorite_side", "favorite_rule_version"}.issubset(entry_rows.columns):
        # Preserve historical entry inference for rows predating provenance.
        side = _identity(_text(row.get("favorite_side")) or _text(row.get("supported_side")) or _text(row.get("side")))
        version = _text(row.get("favorite_rule_version"))
        if side and version and entry_rows["entry_observed_at"].map(_text).ne("").any():
            entry_rows = entry_rows.loc[
                entry_rows["favorite_side"].map(_identity).eq(side)
                & entry_rows["favorite_rule_version"].eq(version)
            ]
    first_lines = entry_rows.get("first_qualified_line", pd.Series(dtype=str))
    entries = entry_rows.get("entry_observed_at", pd.Series(dtype=str))
    first_line = _text(first_lines.iloc[0]) if not first_lines.empty else ""
    entry_observed_at = _text(entries.iloc[0]) if not entries.empty else ""
    if recover_entry and not entry_rows.empty:
        first = entry_rows.iloc[0]
        favorite_side = _text(row.get("favorite_side"))
        if (favorite_side and _identity(first.get("favorite_side")) == _identity(favorite_side)
                and _text(first.get("favorite_rule_version")) == _text(row.get("favorite_rule_version"))):
            verified, observed = tracked_entry_quote(first)
            if verified:
                first_line, entry_observed_at = verified, observed or entry_observed_at
    t20 = scoped[scoped["_at"] <= kickoff - pd.Timedelta(minutes=20)] if pd.notna(kickoff) else pd.DataFrame()
    return {
        "first_qualified": _text(scoped.loc[qualified, "recorded_at"].iloc[0]) if qualified.any() else "",
        "last_qualified": _text(scoped.loc[qualified, "recorded_at"].iloc[-1]) if qualified.any() else "",
        "first_fell_off": _text(scoped.loc[falloffs, "recorded_at"].iloc[0]) if falloffs.any() else "",
        "last_fell_off": _text(scoped.loc[falloffs, "recorded_at"].iloc[-1]) if falloffs.any() else "",
        "episodes": str(int(episode_starts.sum())),
        "requalifications": str(max(0, int(episode_starts.sum()) - 1)),
        "falloffs": str(int(falloffs.sum())),
        "qualifying_captures": str(int(qualified.sum())),
        "whipsaw_ever": "yes" if whipsaw.any() else "no",
        "pathway": _text(pathways.iloc[-1]) if not pathways.empty else "",
        "first_qualified_line": first_line,
        "entry_observed_at": entry_observed_at,
        "history_available": "yes",
        "final_state": _text(states.iloc[-1]) or "unknown",
        "t20_state": _text(t20.iloc[-1].get("favorite_state")) if not t20.empty else "not_observed",
    }


def _lifecycle(history: pd.DataFrame, row: pd.Series, kickoff: pd.Timestamp) -> tuple[str, str]:
    if history.empty:
        return _text(row.get("favorite_first_qualified_at")), "no"
    mask = pd.Series(True, index=history.index)
    for column in ("sport", "game_id", "market_display"):
        if column not in history:
            return _text(row.get("favorite_first_qualified_at")), "no"
        mask &= history[column].astype(str).eq(_text(row.get(column)))
    scoped = history[mask].copy()
    if scoped.empty:
        return _text(row.get("favorite_first_qualified_at")), "no"
    scoped["_at"] = pd.to_datetime(scoped.get("recorded_at", ""), errors="coerce", utc=True)
    if pd.notna(kickoff):
        scoped = scoped[scoped["_at"].notna() & (scoped["_at"] <= kickoff)]
    scoped = scoped.sort_values("_at", kind="mergesort")
    qualified = scoped[scoped.get("favorite_state", "").astype(str).eq("qualified")]
    first = _text(qualified.iloc[0].get("recorded_at")) if not qualified.empty else _text(row.get("favorite_first_qualified_at"))
    fell_off = False
    if not qualified.empty:
        first_at = qualified.iloc[0]["_at"]
        fell_off = bool(((scoped["_at"] > first_at) & scoped.get("favorite_state", "").astype(str).eq("not_qualified")).any())
    return first, "yes" if fell_off else "no"


def _frozen_records(frozen: pd.DataFrame, history: pd.DataFrame, source_name: str) -> list[dict]:
    records: list[dict] = []
    if frozen.empty:
        return records
    ordered = frozen.loc[~frozen.get("freeze_method", pd.Series("", index=frozen.index)).eq(ufc.CONTEXT)].copy()
    ordered["_frozen"] = _utc_series(ordered.get("frozen_at_utc", ""), ordered.index)
    ordered["_state_at"] = _utc_series(ordered.get("final_pregame_state_at_utc", ""), ordered.index)
    ordered["_kickoff"] = _utc_series(ordered.get("kickoff_iso", ""), ordered.index)
    key_columns = [column for column in ("sport", "game_id", "market_display") if column in ordered]
    if len(key_columns) != 3:
        return records
    # The classification is selected by the timestamp of the Red Fox source
    # state, never by the wall-clock time at which this service happens to run.
    # Rows without a proved source time, or sourced after kickoff, are excluded.
    ordered = ordered[
        ordered["_state_at"].notna()
        & ordered["_kickoff"].notna()
        & (ordered["_state_at"] <= ordered["_kickoff"])
    ].copy()
    order = ["_state_at", "_frozen"]
    if "favorite_origin" in ordered:
        ordered["_manual_handoff"] = ordered["favorite_origin"].eq("manual") & ordered["freeze_method"].isin({MANUAL_HANDOFF, ufc.HANDOFF})
        order.insert(0, "_manual_handoff")
    ordered = ordered.sort_values(order, kind="mergesort").drop_duplicates(key_columns, keep="last")
    for _, row in ordered.iterrows():
        kickoff = pd.to_datetime(row.get("kickoff_iso", ""), errors="coerce", utc=True)
        if pd.isna(kickoff) or kickoff < SUPPORTED_SIDE_VALID_FROM:
            continue
        nhl_replacement = official_nhl_favorite(row)
        if nhl_replacement and (row.get("freeze_method") != OFFICIAL_HANDOFF or not _truthy(row.get("red_fox_favorite"))):
            continue
        ufc_proof = ufc.row_proof(row) if row.get("sport") == "ufc" else {}
        ufc_verified = row.get("freeze_method") == ufc.HANDOFF
        if ufc_verified != bool(ufc_proof):
            continue
        manual = (
            row.get("favorite_origin") == "manual" and row.get("favorite_rule_version") == MANUAL_VERSION
            and row.get("freeze_method") in {MANUAL_HANDOFF, ufc.HANDOFF}
        )
        if row.get("favorite_origin") == "manual" and not manual:
            continue
        published_supported = _text(row.get("supported_side"))
        extra = (cfb_spread_favorite(row) or nba_public_favorite(row) or nhl_replacement
                 or mlb_static_favorite(row) or ufc_verified or manual) and _truthy(row.get("red_fox_favorite"))
        supported = _text(row.get("favorite_side")) if extra else published_supported
        if not supported:
            continue
        side_record = _side_record(row, supported)
        if not side_record:
            # A direction without one exact source-side match is not auditable.
            continue
        market = _text(row.get("market_display")).upper()
        line, price = _line_and_price(market, supported, side_record.get("current_line"))
        open_line, open_price = _line_and_price(market, supported, side_record.get("open_line"))
        lifecycle = _tracking_lifecycle(history, row, kickoff, recover_entry=True)
        first_qualified = lifecycle["first_qualified"] or _text(row.get("favorite_first_qualified_at"))
        fell_off = (
            "yes" if int(_number(lifecycle["falloffs"]) or 0) > 0 else "no"
        ) if lifecycle["history_available"] == "yes" else "unknown"
        late_invalidated = _truthy(row.get("favorite_late_invalidated"))
        originally_qualified = _truthy(row.get("favorite_originally_qualified")) or late_invalidated
        favorite = (
            kickoff >= FAVORITE_VALID_FROM
            and _text(row.get("favorite_rule_version")) in FAVORITE_RULE_VERSIONS
            and _truthy(row.get("red_fox_favorite"))
            and _identity(row.get("favorite_side")) == _identity(supported)
            and (_text(row.get("game_id")), market) not in FAVORITE_KPI_EXCLUSIONS
        )
        try:
            market_sides = [item for item in json.loads(_text(row.get("market_sides")) or "[]") if isinstance(item, dict)]
        except (TypeError, ValueError, json.JSONDecodeError):
            market_sides = []
        pathway = _text(row.get("favorite_pathway")) or lifecycle["pathway"] or _candidate_pathway(side_record, market_sides)
        tracked_decision = lifecycle["first_qualified_line"]
        decision_line, decision_price = _line_and_price(market, supported, tracked_decision)
        if manual:
            quote_at = pd.to_datetime(row.get("manual_quote_at"), errors="coerce", utc=True)
            if (not favorite or not _text(row.get("manual_provider_event_id"))
                    or pd.isna(quote_at) or quote_at > row["_state_at"] or row["_state_at"] >= kickoff
                    or _american_profit(row.get("manual_entry_price")) is None):
                continue
            first_qualified = _text(row.get("favorite_first_qualified_at"))
            decision_line, decision_price = _text(row.get("manual_entry_line")), _text(row.get("manual_entry_price"))
            if market in {"SPREAD", "TOTAL"} and _number(decision_line) is None:
                continue
        if nhl_replacement:
            try:
                proof = json.loads(row.get("favorite_supporting_evidence") or "{}")
            except (TypeError, ValueError):
                continue
            if (not favorite or not decision_price or not lifecycle["entry_observed_at"]
                    or tracked_decision != proof.get("entry_line")
                    or lifecycle["entry_observed_at"] != proof.get("entry_source_at")):
                continue
        if (nba_public_favorite(row) or mlb_static_favorite(row)) and favorite and (not decision_price or not lifecycle["entry_observed_at"]):
            continue
        if ufc_verified and not manual:
            entry = ufc.stamp(lifecycle["entry_observed_at"])
            if (not favorite or not decision_price or pd.isna(entry) or entry >= kickoff
                    or ufc_proof.get("bout_id") != row.get("score_provider_event_id")):
                continue
        if not tracked_decision and favorite and not manual:
            decision_line, decision_price = line, price
        decision_source = ("owner_selected_quote" if manual else
                           "first_qualified_capture" if tracked_decision else ("final_pregame_fallback" if favorite else ""))
        recorded_side = _side_at_decision_line(supported, market, decision_line)
        movement_start = _number(open_price if market == "MONEYLINE" else open_line)
        movement_end = _number(price if market == "MONEYLINE" else line)
        line_move = "" if movement_start is None or movement_end is None else str(round(movement_end - movement_start, 4))
        movement_magnitude = _number(side_record.get("price_move_pct")) if market == "MONEYLINE" else _number(line_move)
        contradiction_state = _text(row.get("favorite_cross_market_state")) or (
            "mismatch" if _truthy(row.get("cross_market_mismatch")) or _truthy(row.get("cross_market_opener_mismatch_verified"))
            else "clear"
        )
        favorite_reason = _text(row.get("favorite_reason")).lower()
        locked_decision = (
            "favorite" if "official favorite locked at t-20" in favorite_reason
            else "non_favorite" if "official non-favorite state locked at t-20" in favorite_reason
            else "not_recorded"
        )
        whipsaw_state = _text(row.get("favorite_whipsaw_state"))
        if not whipsaw_state:
            whipsaw_state = (
                "recovered" if _truthy(side_record.get("whipsaw_recovered"))
                else "active_or_partial" if _text(side_record.get("path")).lower() == "whipsaw"
                else "none"
            )
        whipsaw_ever = "yes" if lifecycle["whipsaw_ever"] == "yes" or whipsaw_state != "none" else "no"
        frozen_at = _text(row.get("frozen_at_utc"))
        state_at = row["_state_at"].isoformat()
        first_minutes = _minutes_before(row.get("kickoff_iso"), first_qualified)
        final_minutes = _minutes_before(row.get("kickoff_iso"), state_at)
        source_payload = "|".join([
            _text(row.get("sport")), _text(row.get("game_id")), market, supported,
            _text(side_record.get("current_line")), frozen_at, _text(row.get("favorite_snapshot_id")),
        ])
        ledger_id = hashlib.sha256("|".join(source_payload.split("|")[:3]).encode("utf-8")).hexdigest()[:20]
        records.append({
            "ledger_id": ledger_id,
            "event_id": _text(row.get("game_id")), "game": _text(row.get("game")),
            "sport": _text(row.get("sport")), "scheduled_start": _text(row.get("kickoff_iso")),
            "market": market, "side": recorded_side, "open_line": open_line, "open_price": open_price,
            "decision_line": decision_line, "decision_price": decision_price,
            "decision_line_source": decision_source,
            "decision_quote_observed_at": (
                _text(row.get("manual_quote_at")) if manual else lifecycle["entry_observed_at"] if tracked_decision else
                _text(side_record.get("source_latest_at")) if favorite else ""
            ),
            "final_pregame_line": line,
            "final_pregame_price": price,
            "line_move_from_open": line_move,
            "market_read": _text(side_record.get("reaction")),
            "market_read_detail": _text(side_record.get("anomaly_chips")) or _text(side_record.get("reaction")),
            "market_path": _text(side_record.get("path")),
            "bets_pct": _text(side_record.get("bets_pct")),
            "money_pct": _text(side_record.get("money_pct")),
            "bets_pct_band": _percent_band(side_record.get("bets_pct")),
            "money_pct_band": _percent_band(side_record.get("money_pct")),
            "movement_magnitude": "" if movement_magnitude is None else str(round(abs(movement_magnitude), 4)),
            "movement_magnitude_band": _movement_band(movement_magnitude),
            "favorite_cross_market_state": contradiction_state,
            "observation_count": _text(side_record.get("observation_count")),
            "line_direction_changes": _text(side_record.get("line_dir_changes")),
            "return_toward_open": "yes" if _truthy(side_record.get("return_toward_open")) else "no",
            "active_worsening_reversal": "yes" if _truthy(side_record.get("active_worsening_reversal")) else "no",
            "whipsaw_state": whipsaw_state, "whipsaw_ever": whipsaw_ever,
            "supported_side": published_supported, "favorite_qualified": "yes" if favorite else "no",
            "favorite_origin": "manual" if manual else "", "manual_revision": _text(row.get("manual_revision")) if manual else "",
            "manual_provider_event_id": _text(row.get("manual_provider_event_id")) if manual else "",
            "favorite_rule_version": _text(row.get("favorite_rule_version")),
            "favorite_originally_qualified": "yes" if originally_qualified or favorite else "no",
            "favorite_late_invalidated": "yes" if late_invalidated else "no",
            "favorite_late_invalidated_at": _text(row.get("favorite_late_invalidated_at")),
            "favorite_late_invalidated_reason": _text(row.get("favorite_late_invalidated_reason")),
            "favorite_pathway": pathway,
            "candidate_decision": "late_invalidated" if late_invalidated else ("accepted" if favorite else "rejected"),
            "candidate_decision_reason": (
                _text(row.get("favorite_late_invalidated_reason"))
                if late_invalidated else _candidate_reason(row, side_record, pathway, favorite)
            ),
            "price_band": _price_band(price) if market == "MONEYLINE" else "not_applicable",
            "spread_band": _spread_band(line, market),
            "favorite_first_qualified_at": first_qualified,
            "favorite_final_qualified_at": _text(row.get("favorite_final_qualified_at")) if favorite else "",
            "favorite_first_fell_off_at": lifecycle["first_fell_off"],
            "favorite_last_fell_off_at": lifecycle["last_fell_off"],
            "favorite_fell_off_before_kickoff": fell_off,
            "favorite_ever_fell_off": fell_off,
            "favorite_final_state": "late_invalidated" if late_invalidated else ("qualified" if favorite else "not_qualified"),
            "favorite_t20_state": lifecycle["t20_state"],
            "favorite_locked_decision": locked_decision,
            "qualification_episode_count": lifecycle["episodes"],
            "requalification_count": lifecycle["requalifications"],
            "falloff_count": lifecycle["falloffs"],
            "qualifying_capture_count": lifecycle["qualifying_captures"],
            "minutes_first_qualified_before_start": first_minutes,
            "minutes_final_state_before_start": final_minutes,
            "first_qualified_timing_band": _timing_band(first_minutes),
            "final_state_timing_band": _timing_band(final_minutes),
            "final_pregame_state_at_utc": state_at,
            "final_pregame_frozen_at": frozen_at,
            "classification_correction": "yes" if _truthy(row.get("classification_correction")) else "no",
            "classification_corrected_at_utc": _text(row.get("classification_corrected_at_utc")),
            "classification_correction_reason": _text(row.get("classification_correction_reason")),
            "classification_original_publication": _text(row.get("classification_original_publication")) or "published",
            "final_score": "", "market_result": "", "grade": "",
            "stake_units": "", "net_profit_units": "", "roi": "",
            "closing_line": "" if manual else _text(row.get("closing_line")) or line,
            "closing_price": "" if manual else _text(row.get("closing_price")) or price,
            "closing_line_source": "unavailable" if manual else _closing_source(row, "closing_line", line),
            "closing_price_source": "unavailable" if manual else _closing_source(row, "closing_price", price),
            "closing_line_observed_at": "" if manual else _closing_observed_at(row, "closing_line", line, side_record),
            "closing_price_observed_at": "" if manual else _closing_observed_at(row, "closing_price", price, side_record),
            "clv": _text(row.get("clv")), "clv_method": "retained_source" if _text(row.get("clv")) else "",
            "clv_available": "no",
            "score_provider": "espn_ufc_bout" if ufc_verified else "", "score_source": "",
            "score_provider_event_id": ufc_proof.get("bout_id", ""),
            "score_resolved_at": "", "graded_at_utc": "", "grade_evidence_hash": "",
            "freeze_source": source_name,
            "freeze_method": _text(row.get("freeze_method")) or "audited_latest_state_at_or_before_start",
            "source_snapshot_id": _text(row.get("favorite_snapshot_id")),
            "source_hash": hashlib.sha256(source_payload.encode("utf-8")).hexdigest(),
            "audit_status": "awaiting_final_score",
        })
    return records


def _closing_source(row: pd.Series, field: str, fallback: str) -> str:
    if _text(row.get(field)):
        return _text(row.get(f"{field}_source")) or "retained_close_unverified"
    return "final_pregame_fallback" if _text(fallback) else "unavailable"


def _closing_observed_at(row: pd.Series, field: str, fallback: str, side: dict) -> str:
    if _text(row.get(field)):
        return _text(row.get(f"{field}_observed_at"))
    return _text(side.get("source_latest_at")) if _text(fallback) else ""


def _apply_retroactive_favorite_inclusions(frame: pd.DataFrame) -> pd.DataFrame:
    """Repair proved system omissions from retained pregame source evidence."""
    result = frame.copy()
    if result.empty or not {"game_id", "market_display"}.issubset(result.columns):
        return result
    for index, row in result.iterrows():
        key = (_text(row.get("game_id")), _text(row.get("market_display")).upper())
        correction = RETROACTIVE_FAVORITE_INCLUSIONS.get(key)
        if not correction:
            continue
        try:
            sides = [
                item for item in json.loads(_text(row.get("market_sides")) or "[]")
                if isinstance(item, dict)
            ]
        except (TypeError, ValueError, json.JSONDecodeError):
            sides = []
        team = correction["team"]
        side = next((item for item in sides if _identity(item.get("flagged_side")) == _identity(team)), None)
        if side is None:
            continue
        evidence = {
            "reaction": _text(side.get("reaction")),
            "path": _text(side.get("path")),
            "bets_pct": _number(side.get("bets_pct")),
            "money_pct": _number(side.get("money_pct")),
            "open_line": _text(side.get("open_line")),
            "current_line": _text(side.get("current_line")),
            "observation_count": int(_number(side.get("observation_count")) or 0),
            "cross_market": "confirmation",
            "retroactive_system_correction": True,
        }
        result.at[index, "supported_side"] = team
        result.at[index, "red_fox_favorite"] = "true"
        result.at[index, "favorite_side"] = team
        result.at[index, "favorite_pathway"] = "low_ticket_high_money_contrarian"
        result.at[index, "favorite_rule_version"] = "red_fox_favorite_v8"
        result.at[index, "favorite_first_qualified_at"] = correction["first_qualified_at"]
        result.at[index, "favorite_final_qualified_at"] = correction["final_qualified_at"]
        result.at[index, "favorite_state"] = "qualified"
        result.at[index, "favorite_final_market_read"] = _text(side.get("anomaly_chips")) or _text(side.get("reaction"))
        result.at[index, "favorite_supporting_evidence"] = json.dumps(evidence, separators=(",", ":"))
        result.at[index, "favorite_cross_market_state"] = "confirmation"
        result.at[index, "favorite_originally_qualified"] = "true"
        result.at[index, "favorite_reason"] = correction["reason"]
        result.at[index, "classification_correction"] = "true"
        result.at[index, "classification_corrected_at_utc"] = correction["corrected_at"]
        result.at[index, "classification_correction_reason"] = correction["reason"]
        result.at[index, "classification_original_publication"] = "missed"
        result.at[index, "freeze_method"] = "retroactive_system_correction_from_retained_pregame_state"
    return result


def _score_map(scores: pd.DataFrame) -> dict[str, pd.Series]:
    if scores.empty or "game_id" not in scores:
        return {}
    # The immutable final-score history is concatenated first; frozen live rows
    # are fallback evidence and may not overwrite it during routine rebuilds.
    return {str(row.get("game_id", "")): row for _, row in scores.drop_duplicates("game_id", keep="first").iterrows()}


def _scores_from_frozen(frozen: pd.DataFrame) -> pd.DataFrame:
    """Use already-retained final live scores; this performs no API call."""
    if frozen.empty or "score_state" not in frozen:
        return pd.DataFrame()
    final = frozen[frozen["score_state"].astype(str).str.lower().eq("post")].copy()
    if "favorite_origin" in final:
        final = final.assign(_manual_score=final["favorite_origin"].eq("manual")).sort_values("_manual_score", kind="stable")
    records = []
    for _, row in final.drop_duplicates("game_id", keep="last").iterrows():
        game = _text(row.get("game"))
        if " @ " not in game:
            continue
        away, home = game.split(" @ ", 1)
        source = MLB_SCORE_SOURCE if (_text(row.get("sport")).lower() == "mlb" and _text(row.get("score_provider")) == "mlb"
                                  and _text(row.get("score_match_state")) == "matched") else "live_recent.csv"
        records.append({
            "game_id": _text(row.get("game_id")), "team1": away,
            "team1_score": _text(row.get("score_away")), "team2": home,
            "team2_score": _text(row.get("score_home")),
            "score_provider": _text(row.get("score_provider")) or "retained_live_recent",
            "score_source": source,
            "score_provider_event_id": _text(row.get("score_provider_event_id")),
            "manual_score_status": _text(row.get("score_status_code")) if row.get("favorite_origin") == "manual" else "",
            "resolved_at_utc": _text(row.get("score_completed_at_utc")) or _text(row.get("score_updated_at_utc")),
        })
    return pd.DataFrame(records)


def _fetch_ufc_results(ledger: pd.DataFrame, now=None) -> pd.DataFrame:
    """Resolve completed UFC moneylines from ESPN's bout-level scoreboard.

    Team score feeds do not represent combat sports. Match the two retained
    fighter identities to a completed ESPN bout and preserve its explicit
    winner instead of manufacturing numeric scores.
    """
    if ledger.empty:
        return pd.DataFrame()
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    starts = pd.to_datetime(ledger.get("scheduled_start", ""), errors="coerce", utc=True)
    pending = ledger.loc[
        ledger.get("sport", pd.Series("", index=ledger.index)).astype(str).str.lower().eq("ufc")
        & ledger.get("market", pd.Series("", index=ledger.index)).astype(str).str.upper().eq("MONEYLINE")
        & ~ledger.get("freeze_method", pd.Series("", index=ledger.index)).eq(ufc.HANDOFF)
        & ~ledger.get("grade", pd.Series("", index=ledger.index)).astype(str).isin({"W", "L", "Push"})
        & ledger.get("event_id", pd.Series("", index=ledger.index)).astype(str).str.fullmatch(r"\d+")
        & starts.notna() & starts.le(current)
        & starts.ge(current - pd.Timedelta(days=21))
    ].copy()
    if pending.empty:
        return pd.DataFrame()
    pending["_date"] = starts.loc[pending.index].dt.strftime("%Y%m%d")
    records: list[dict] = []
    for date, dated in pending.groupby("_date", sort=False):
        try:
            response = requests.get(
                "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard",
                params={"dates": date}, timeout=12,
            )
            response.raise_for_status()
            events = response.json().get("events", [])
        except (requests.RequestException, ValueError, TypeError):
            continue
        bouts = []
        for event in events:
            for bout in event.get("competitions", []):
                status = bout.get("status", {}).get("type", {})
                competitors = bout.get("competitors", [])
                if not status.get("completed") or len(competitors) != 2:
                    continue
                names = [
                    _text(item.get("athlete", {}).get("displayName") or item.get("athlete", {}).get("fullName"))
                    for item in competitors
                ]
                winners = [name for name, item in zip(names, competitors) if item.get("winner") is True]
                if len(winners) != 1 or not all(names):
                    continue
                winner = winners[0]
                loser = names[1] if _identity(names[0]) == _identity(winner) else names[0]
                bouts.append((frozenset(map(_identity, names)), winner, loser, _text(bout.get("id"))))
        for _, row in dated.iterrows():
            fighters = re.split(r"\s+vs\.?\s+", _text(row.get("game")), maxsplit=1, flags=re.I)
            if len(fighters) != 2:
                continue
            target = frozenset(map(_identity, fighters))
            matches = [bout for bout in bouts if bout[0] == target]
            if len(matches) != 1:
                continue
            _, winner, loser, provider_id = matches[0]
            records.append({
                "game_id": _text(row.get("event_id")), "winner": winner, "loser": loser,
                "result": f"{winner} defeated {loser}", "score_provider": "espn_ufc_scoreboard",
                "score_source": "ESPN UFC scoreboard", "score_provider_event_id": provider_id,
                "resolved_at_utc": current.isoformat(),
            })
    return pd.DataFrame(records)


def _refresh_ufc_result_history(ledger: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    path = data_dir / "ufc_results_history.csv"
    history = _read_csv(path)
    fetched = _fetch_ufc_results(ledger)
    if fetched.empty:
        return history
    updated = pd.concat([history, fetched], ignore_index=True, sort=False) if not history.empty else fetched
    updated = updated.drop_duplicates("game_id", keep="last")
    _atomic_csv(updated, path)
    return updated


def _american_profit(price: object) -> float | None:
    odds = _number(price)
    if odds is None or -100 < odds < 100 or odds == 0:
        return None
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def _implied_probability(price: object) -> float | None:
    odds = _number(price)
    if odds is None or -100 < odds < 100 or odds == 0:
        return None
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def _attach_clv(result: pd.DataFrame, index: object, row: pd.Series) -> None:
    result.at[index, "clv_available"] = "no"
    market = _text(row.get("market")).upper()
    if market not in {"MONEYLINE", "SPREAD"}:
        return
    field = "closing_price" if market == "MONEYLINE" else "closing_line"
    if _text(row.get(f"{field}_source")) != "verified_closing_capture":
        return
    entry, close, start = (
        pd.to_datetime(row.get(column), errors="coerce", utc=True)
        for column in ("decision_quote_observed_at", f"{field}_observed_at", "scheduled_start")
    )
    if not (pd.notna(entry) and pd.notna(close) and pd.notna(start) and entry < close <= start):
        return
    if market == "MONEYLINE":
        decision = _implied_probability(row.get("decision_price"))
        closing = _implied_probability(row.get("closing_price"))
        if decision is not None and closing is not None:
            result.at[index, "clv"] = str(round((closing - decision) * 100.0, 4))
            result.at[index, "clv_method"] = "implied_probability_points"
            result.at[index, "clv_available"] = "yes"
            return
    else:
        decision = _number(row.get("decision_line"))
        closing = _number(row.get("closing_line"))
        if decision is not None and closing is not None:
            result.at[index, "clv"] = str(round(decision - closing, 4))
            result.at[index, "clv_method"] = "side_line_points"
            result.at[index, "clv_available"] = "yes"
            return


def _attach_results(ledger: pd.DataFrame, scores: pd.DataFrame, *, mlb_cutover=None) -> pd.DataFrame:
    result = ledger.copy()
    by_event = _score_map(scores)
    by_fixture = {(str(row.get("game_id", "")), _text(row.get("score_provider_event_id"))): row
                  for _, row in scores.iterrows() if _text(row.get("score_provider_event_id"))}
    verified_bouts = _score_map(scores.loc[scores.get("score_source", pd.Series("", index=scores.index)).eq(ufc.SCORE_SOURCE)])
    graded_at = datetime.now(timezone.utc).isoformat()
    for index, row in result.iterrows():
        _attach_clv(result, index, row)
        score = by_event.get(_text(row.get("event_id")))
        manual = row.get("favorite_origin") == "manual"
        if manual:
            if _text(row.get("grade")) in {"W", "L", "Push", "Void"}:
                continue
            score = by_fixture.get((_text(row.get("event_id")), _text(row.get("manual_provider_event_id"))))
        if mlb_cutover and _text(row.get("sport")).lower() == "mlb":
            scheduled = mlb_timestamp(_text(row.get("scheduled_start")))
            if _text(row.get("grade")) in {"W", "L", "Push"} or not scheduled or scheduled < mlb_cutover:
                continue
            if score is None or _text(score.get("score_source")) != MLB_SCORE_SOURCE or not _text(score.get("score_provider_event_id")).isdigit():
                result.at[index, "audit_status"] = "awaiting_verified_mlb_fixture"
                continue
        verified_ufc = row.get("sport") == "ufc" and row.get("freeze_method") == ufc.HANDOFF
        if verified_ufc:
            if _text(row.get("grade")) in {"W", "L", "Push", "Void"}:
                continue
            score = verified_bouts.get(_text(row.get("event_id")))
            if score is None or _text(score.get("score_provider_event_id")) != _text(row.get("score_provider_event_id")):
                result.at[index, "audit_status"] = "awaiting_verified_ufc_bout_result"
                continue
        if score is None:
            if _text(row.get("grade")) in {"W", "L", "Push"} and _text(row.get("final_score")):
                result.at[index, "score_source"] = _text(row.get("score_source")) or "retained_performance_ledger"
                result.at[index, "graded_at_utc"] = _text(row.get("graded_at_utc")) or _text(row.get("score_resolved_at")) or graded_at
                if not _text(row.get("grade_evidence_hash")):
                    result.at[index, "grade_evidence_hash"] = _result_evidence_hash(result.loc[index])
            elif _text(row.get("grade")) in {"W", "L", "Push"}:
                for column in ("grade", "market_result", "stake_units", "net_profit_units", "roi"):
                    result.at[index, column] = ""
                result.at[index, "audit_status"] = "unresolved_missing_score_evidence"
            else:
                result.at[index, "audit_status"] = "awaiting_final_score"
            continue
        winner = _text(score.get("winner"))
        loser = _text(score.get("loser"))
        kind = _text(score.get("ufc_kind")) if verified_ufc else ""
        if kind in {"push", "void"}:
            grade = kind.upper()
        elif manual and _text(score.get("manual_score_status")) in {"STATUS_CANCELED", "STATUS_CANCELLED"}:
            grade = "VOID"
        elif verified_ufc and kind == "winner":
            side = ufc.fighter_key(row.get("side"))
            grade = "WIN" if side == ufc.fighter_key(winner) else "LOSS" if side == ufc.fighter_key(loser) else "UNRESOLVED"
        elif _text(row.get("sport")).lower() == "ufc" and winner and loser:
            side = _identity(row.get("side"))
            grade = "WIN" if side == _identity(winner) else "LOSS" if side == _identity(loser) else "UNRESOLVED"
        else:
            grade_row = {
            "market_display": row.get("market", ""), "action_side": row.get("side", ""),
            "team1": score.get("team1", ""), "team1_score": score.get("team1_score", ""),
            "team2": score.get("team2", ""), "team2_score": score.get("team2_score", ""),
            }
            grade = _grade_action(grade_row)
        if grade == "UNRESOLVED":
            result.at[index, "audit_status"] = "ambiguous_score_or_side_match"
            continue
        team1, team2 = _text(score.get("team1")), _text(score.get("team2"))
        score1, score2 = _text(score.get("team1_score")), _text(score.get("team2_score"))
        result.at[index, "final_score"] = (
            _text(score.get("result")) if verified_ufc or (winner and loser) else f"{team1} {score1} - {team2} {score2}"
        )
        result.at[index, "market_result"] = f"{_text(row.get('side'))}: {grade.title()}"
        result.at[index, "grade"] = {"WIN": "W", "LOSS": "L", "PUSH": "Push", "VOID": "Void"}[grade]
        win_profit = _american_profit(row.get("decision_price"))
        if grade == "VOID":
            for column in ("stake_units", "net_profit_units", "roi"):
                result.at[index, column] = "0"
        elif win_profit is not None:
            result.at[index, "stake_units"] = "1"
            net = win_profit if grade == "WIN" else (-1.0 if grade == "LOSS" else 0.0)
            result.at[index, "net_profit_units"] = f"{round(net, 6):g}"
            result.at[index, "roi"] = f"{round(net, 6):g}"
        result.at[index, "score_provider"] = _text(score.get("score_provider")) or (
            "mlb" if _text(score.get("score_source")) == MLB_SCORE_SOURCE else "espn_final_scores_history"
        )
        result.at[index, "score_source"] = _text(score.get("score_source")) or "final_scores_history.csv"
        result.at[index, "score_provider_event_id"] = _text(score.get("score_provider_event_id"))
        result.at[index, "score_resolved_at"] = _text(score.get("resolved_at_utc"))
        result.at[index, "graded_at_utc"] = _text(row.get("graded_at_utc")) or graded_at
        result.at[index, "grade_evidence_hash"] = _result_evidence_hash(result.loc[index])
        result.at[index, "audit_status"] = "graded"
    for index, row in result.iterrows():
        grade = _text(row.get("grade"))
        win_profit = _american_profit(row.get("decision_price"))
        if grade in {"W", "L", "Push"} and win_profit is not None:
            net = win_profit if grade == "W" else (-1.0 if grade == "L" else 0.0)
            result.at[index, "stake_units"] = "1"
            result.at[index, "net_profit_units"] = f"{round(net, 6):g}"
            result.at[index, "roi"] = f"{round(net, 6):g}"
    return result


def _result_evidence_hash(row: pd.Series) -> str:
    if _text(row.get("grade")) not in {"W", "L", "Push"} or not _text(row.get("final_score")):
        return ""
    payload = "|".join(_text(row.get(column)) for column in (
        "ledger_id", "event_id", "market", "side", "decision_line", "decision_price",
        "final_score", "grade", "score_provider", "score_source",
        "score_provider_event_id", "score_resolved_at",
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cohort_row(rows: pd.DataFrame, dimension: str, segment: str, decision: str) -> dict:
    graded = rows[rows["grade"].isin(["W", "L", "Push"])]
    wins = int(graded["grade"].eq("W").sum())
    losses = int(graded["grade"].eq("L").sum())
    pushes = int(graded["grade"].eq("Push").sum())
    stake = pd.to_numeric(graded.get("stake_units", pd.Series(dtype=str)), errors="coerce")
    profit = pd.to_numeric(graded.get("net_profit_units", pd.Series(dtype=str)), errors="coerce")
    valid_roi = stake.notna() & profit.notna()
    total_stake = float(stake[valid_roi].sum())
    total_profit = float(profit[valid_roi].sum())
    verified = rows.get("clv_available", pd.Series("no", index=rows.index)).eq("yes")
    clv = pd.to_numeric(rows.get("clv", pd.Series("", index=rows.index)).loc[verified], errors="coerce")
    clv_count = int(clv.notna().sum())
    whipsaw = rows.get("whipsaw_ever", pd.Series("no", index=rows.index)).astype(str).eq("yes")
    fell_off = rows.get("favorite_ever_fell_off", pd.Series("unknown", index=rows.index)).astype(str).eq("yes")
    count = len(rows)
    return {
        "dimension": dimension, "segment": segment, "candidate_decision": decision,
        "candidates": count, "graded": len(graded), "wins": wins, "losses": losses,
        "pushes": pushes,
        "win_rate_excluding_pushes": round(wins / (wins + losses), 6) if wins + losses else "",
        "stake_units": round(total_stake, 6) if valid_roi.any() else "",
        "net_profit_units": round(total_profit, 6) if valid_roi.any() else "",
        "roi": round(total_profit / total_stake, 6) if total_stake else "",
        "clv_available": clv_count,
        "clv_coverage": round(clv_count / count, 6) if count else "",
        "average_clv": round(float(clv.mean()), 6) if clv_count else "",
        "whipsaw_candidates": int(whipsaw.sum()),
        "whipsaw_rate": round(float(whipsaw.mean()), 6) if count else "",
        "fell_off_candidates": int(fell_off.sum()),
        "falloff_rate": round(float(fell_off.mean()), 6) if count else "",
        "sample_status": (
            "TOO_SMALL" if len(graded) < 10 else
            "EARLY" if len(graded) < 25 else
            "DEVELOPING" if len(graded) < 50 else
            "MATURE"
        ),
    }


def _cohort_kpis(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=COHORT_KPI_COLUMNS)
    records: list[dict] = []
    dimensions = [("all", pd.Series("all", index=ledger.index))]
    dimensions.extend((column, ledger[column].fillna("").astype(str).replace("", "unavailable")) for column in COHORT_DIMENSIONS)
    full = ledger[COHORT_DIMENSIONS].fillna("").astype(str).replace("", "unavailable").agg(" | ".join, axis=1)
    dimensions.append(("full_cohort", full))
    decisions = ledger.get("candidate_decision", pd.Series("rejected", index=ledger.index)).astype(str).replace("", "rejected")
    for dimension, values in dimensions:
        for segment in sorted(values.unique()):
            segment_rows = ledger.loc[values.eq(segment)]
            records.append(_cohort_row(segment_rows, dimension, segment, "all"))
            for decision in sorted(decisions.loc[segment_rows.index].unique()):
                scoped = segment_rows.loc[decisions.loc[segment_rows.index].eq(decision)]
                if not scoped.empty:
                    records.append(_cohort_row(scoped, dimension, segment, decision))
    return pd.DataFrame(records, columns=COHORT_KPI_COLUMNS)


def _exports(ledger: pd.DataFrame, data_dir: Path) -> None:
    combined = ledger.reindex(columns=LEDGER_COLUMNS).sort_values(["scheduled_start", "event_id", "market"], kind="mergesort")
    _atomic_csv(combined, data_dir / EXPORT_FILES["combined"])
    _atomic_csv(combined, data_dir / EXPORT_FILES["supported"])
    favorites = combined[combined["favorite_qualified"].eq("yes")].copy()
    _atomic_csv(favorites, data_dir / EXPORT_FILES["favorites"])
    _atomic_csv(combined, data_dir / EXPORT_FILES["candidate_audit"])
    _atomic_csv(_cohort_kpis(combined), data_dir / EXPORT_FILES["cohort_kpis"])


def _update_ncaaf_shadow_results(ledger: pd.DataFrame, data_dir: Path) -> None:
    """Attach outcomes to prospective NCAAF shadow rows without reclassifying history."""
    path = data_dir / "red_fox_favorite_shadow.csv"
    shadow = _read_csv(path)
    if shadow.empty or ledger.empty:
        return
    result_columns = ["closing_spread", "game_result", "grade", "units_won_lost", "clv"]
    for column in result_columns:
        if column not in shadow:
            shadow[column] = ""
    graded = ledger.loc[
        ledger.get("sport", pd.Series("", index=ledger.index)).astype(str).str.lower().isin({"ncaaf", "cfb"})
        & ledger.get("grade", pd.Series("", index=ledger.index)).astype(str).isin({"W", "L", "Push"})
    ].copy()
    if graded.empty:
        return
    # The former row-by-row DataFrame filtering made this two-minute worker run
    # continuously as shadow history grew. Build constant-time lookup maps and
    # touch only college-football shadow rows.
    graded["_event_market"] = list(zip(
        graded["event_id"].astype(str), graded["market"].astype(str).str.upper(),
    ))
    graded["_side_identity"] = graded["side"].map(_identity)
    exact = {
        (row["_event_market"], row["_side_identity"]): row
        for _, row in graded.iterrows()
    }
    fallback = {row["_event_market"]: row for _, row in graded.iterrows()}
    scoped = shadow.get("sport", pd.Series("", index=shadow.index)).astype(str).str.lower().isin({"ncaaf", "cfb"})
    for index in shadow.index[scoped]:
        key = (
            _text(shadow.at[index, "game_id"]),
            _text(shadow.at[index, "market_display"]).upper(),
        )
        matched = exact.get((key, _identity(shadow.at[index, "candidate_side"])))
        if matched is None:
            matched = fallback.get(key)
        if matched is None:
            continue
        shadow.at[index, "closing_spread"] = matched.get("closing_line", "")
        shadow.at[index, "game_result"] = matched.get("market_result", "") or matched.get("final_score", "")
        shadow.at[index, "grade"] = matched.get("grade", "")
        shadow.at[index, "units_won_lost"] = matched.get("net_profit_units", "")
        shadow.at[index, "clv"] = matched.get("clv", "")
    _atomic_csv(shadow, path)


def update_performance_ledger(
    data_dir: Path | str = "data",
    frozen_sources: list[Path] | None = None,
    attach_results: bool = True,
) -> dict:
    """Ingest only unseen frozen rows, attach known results, and write admin exports."""
    data_dir = Path(data_dir)
    ledger_path = data_dir / "performance_ledger.csv"
    history = _read_csv(data_dir / "red_fox_favorite_tracking.csv")
    ledger = _read_csv(ledger_path)
    for column in LEDGER_COLUMNS:
        if column not in ledger:
            ledger[column] = ""
    sources = frozen_sources or [data_dir / "live_recent.csv"]
    candidates = []
    source_frames = []
    for source in sources:
        frame = _apply_retroactive_favorite_inclusions(_read_csv(Path(source)))
        source_frames.append(frame)
        candidates.extend(_frozen_records(frame, history, str(Path(source))))
    existing_ids = set(ledger.get("ledger_id", pd.Series(dtype=str)).astype(str))
    official_methods = {OFFICIAL_HANDOFF, ufc.HANDOFF, MANUAL_HANDOFF}
    official_handoff_ids = {record["ledger_id"] for record in candidates if record["freeze_method"] in official_methods}
    def event_key(record):
        return record["sport"], record["event_id"]
    official_events = {event_key(r) for r in ledger.loc[ledger["favorite_qualified"].eq("yes")].to_dict("records")}
    manual_events = {event_key(r) for r in candidates if r["favorite_origin"] == "manual"} | {
        event_key(r) for r in ledger.loc[ledger["favorite_origin"].eq("manual")].to_dict("records")
    }
    new_records = []
    seen_nhl_replacements = set()
    for record in candidates:
        identity = record["ledger_id"]
        if identity in existing_ids or identity in seen_nhl_replacements:
            continue
        event = event_key(record)
        if event in manual_events and record["favorite_qualified"] == "yes":
            if record["favorite_origin"] != "manual" or event in official_events:
                continue
            official_events.add(event)
        if record["freeze_method"] in official_methods:
            seen_nhl_replacements.add(identity)
        elif identity in official_handoff_ids:
            continue
        new_records.append(record)
    if new_records:
        ledger = pd.concat([ledger, pd.DataFrame(new_records)], ignore_index=True, sort=False)
    if not ledger.empty:
        # Migrate retained rows to the same immutable recording contract.  This
        # does not change Supported Side or the final pregame/closing line.
        prior_side = ledger["side"].copy()
        ledger["side"] = ledger.apply(
            lambda row: _side_at_decision_line(row.get("side"), row.get("market"), row.get("decision_line")),
            axis=1,
        )
        relabeled = prior_side.ne(ledger["side"]) & ledger["grade"].isin(["W", "L", "Push"])
        ledger.loc[relabeled, "market_result"] = ledger.loc[relabeled].apply(
            lambda row: f"{row.get('side', '')}: "
            f"{('Win' if row.get('grade') == 'W' else 'Loss' if row.get('grade') == 'L' else 'Push')}",
            axis=1,
        )
        excluded = pd.Series(
            [
                (_text(row.get("event_id")), _text(row.get("market")).upper())
                in FAVORITE_KPI_EXCLUSIONS
                for _, row in ledger.iterrows()
            ],
            index=ledger.index,
        )
        ledger.loc[excluded, "favorite_qualified"] = "no"
        ledger.loc[excluded, "favorite_final_qualified_at"] = ""
        for index in ledger.index[excluded]:
            exclusion_key = (_text(ledger.at[index, "event_id"]), _text(ledger.at[index, "market"]).upper())
            ledger.at[index, "candidate_decision"] = "rule_invalidated"
            ledger.at[index, "candidate_decision_reason"] = FAVORITE_KPI_EXCLUSIONS[exclusion_key]
            ledger.at[index, "classification_correction"] = "yes"
            ledger.at[index, "classification_correction_reason"] = FAVORITE_KPI_EXCLUSIONS[exclusion_key]
            ledger.at[index, "classification_original_publication"] = "published"
        blank_late = ledger["favorite_late_invalidated"].astype(str).str.strip().eq("")
        ledger.loc[blank_late, "favorite_late_invalidated"] = "no"
        blank_original = ledger["favorite_originally_qualified"].astype(str).str.strip().eq("")
        ledger.loc[blank_original, "favorite_originally_qualified"] = ledger.loc[blank_original].apply(
            lambda row: "yes" if _truthy(row.get("favorite_qualified")) or _truthy(row.get("favorite_late_invalidated")) else "no",
            axis=1,
        )
        blank_decision = ledger["candidate_decision"].astype(str).str.strip().eq("")
        late_invalidated = ledger["favorite_late_invalidated"].map(_truthy)
        ledger.loc[blank_decision & late_invalidated, "candidate_decision"] = "late_invalidated"
        undecided = blank_decision & ~late_invalidated
        ledger.loc[undecided, "candidate_decision"] = ledger.loc[undecided, "favorite_qualified"].map(
            lambda value: "accepted" if _truthy(value) else "rejected"
        )
        blank_reason = ledger["candidate_decision_reason"].astype(str).str.strip().eq("")
        ledger.loc[blank_reason & ledger["candidate_decision"].eq("accepted"), "candidate_decision_reason"] = (
            "Accepted by final pregame Favorite rules."
        )
        ledger.loc[blank_reason & ledger["candidate_decision"].eq("rejected"), "candidate_decision_reason"] = (
            "Rejected; detailed gate evidence was not retained in this legacy row."
        )
        ledger.loc[blank_reason & ledger["candidate_decision"].eq("late_invalidated"), "candidate_decision_reason"] = (
            ledger.loc[blank_reason & ledger["candidate_decision"].eq("late_invalidated"), "favorite_late_invalidated_reason"]
            .replace("", "Originally qualified, then suppressed by a late safety invalidation.")
        )
        # A Live & Recent handoff can retain an older qualification timestamp
        # even when tracking subsequently fell off and requalified before
        # kickoff.  Judge the handoff by the latest explicit pre-kickoff state,
        # not by comparing its retained timestamp with an earlier falloff.
        latest_tracking_state = pd.Series(
            [
                _tracking_lifecycle(
                    history,
                    pd.Series({
                        "sport": row.get("sport", ""),
                        "game_id": row.get("event_id", ""),
                        "market_display": row.get("market", ""),
                    }),
                    pd.to_datetime(row.get("scheduled_start", ""), errors="coerce", utc=True),
                )["final_state"]
                for _, row in ledger.iterrows()
            ],
            index=ledger.index,
        )
        stale_handoff = (
            ledger.get("freeze_method", pd.Series("", index=ledger.index)).astype(str)
            .eq("favorite_tracking_last_qualified_at_or_before_start")
            & latest_tracking_state.eq("not_qualified")
        )
        restored_handoff = (
            ledger.get("freeze_method", pd.Series("", index=ledger.index)).astype(str)
            .eq("favorite_tracking_last_qualified_at_or_before_start")
            & latest_tracking_state.eq("qualified")
            & ledger["candidate_decision"].eq("stale_handoff_invalidated")
            & ~excluded
            & ~late_invalidated
        )
        ledger.loc[restored_handoff, "favorite_qualified"] = "yes"
        ledger.loc[restored_handoff, "candidate_decision"] = "accepted"
        ledger.loc[restored_handoff, "candidate_decision_reason"] = (
            "Accepted: the latest explicit pre-kickoff tracking state was qualified."
        )
        ledger.loc[stale_handoff, "favorite_qualified"] = "no"
        ledger.loc[stale_handoff, "candidate_decision"] = "stale_handoff_invalidated"
        ledger.loc[stale_handoff, "candidate_decision_reason"] = (
            "Excluded: the latest explicit pre-kickoff tracking state was not qualified; "
            "a last-qualified handoff archive must not restore it."
        )
        for index, row in ledger.iterrows():
            lifecycle_row = pd.Series({
                "sport": row.get("sport", ""), "game_id": row.get("event_id", ""),
                "market_display": row.get("market", ""),
            })
            lifecycle = _tracking_lifecycle(
                history, lifecycle_row,
                pd.to_datetime(row.get("scheduled_start", ""), errors="coerce", utc=True),
            )
            if lifecycle["history_available"] == "yes":
                ever_fell_off = "yes" if int(_number(lifecycle["falloffs"]) or 0) > 0 else "no"
                ledger.at[index, "favorite_first_qualified_at"] = lifecycle["first_qualified"] or row.get("favorite_first_qualified_at", "")
                if lifecycle["final_state"] == "qualified":
                    ledger.at[index, "favorite_final_qualified_at"] = lifecycle["last_qualified"]
                ledger.at[index, "favorite_first_fell_off_at"] = lifecycle["first_fell_off"]
                ledger.at[index, "favorite_last_fell_off_at"] = lifecycle["last_fell_off"]
                ledger.at[index, "favorite_fell_off_before_kickoff"] = ever_fell_off
                ledger.at[index, "favorite_ever_fell_off"] = ever_fell_off
                ledger.at[index, "favorite_t20_state"] = lifecycle["t20_state"]
                ledger.at[index, "qualification_episode_count"] = lifecycle["episodes"]
                ledger.at[index, "requalification_count"] = lifecycle["requalifications"]
                ledger.at[index, "falloff_count"] = lifecycle["falloffs"]
                ledger.at[index, "qualifying_capture_count"] = lifecycle["qualifying_captures"]
            else:
                ledger.at[index, "favorite_fell_off_before_kickoff"] = "unknown"
                ledger.at[index, "favorite_ever_fell_off"] = "unknown"
                if not _text(row.get("favorite_t20_state")):
                    ledger.at[index, "favorite_t20_state"] = "unknown"
            if not _text(row.get("favorite_final_state")):
                ledger.at[index, "favorite_final_state"] = (
                    "late_invalidated" if _truthy(row.get("favorite_late_invalidated"))
                    else "qualified" if _truthy(row.get("favorite_qualified"))
                    else "not_qualified"
                )
            if not _text(row.get("favorite_locked_decision")):
                ledger.at[index, "favorite_locked_decision"] = "not_recorded"
            if not _text(row.get("decision_line")) and not _text(row.get("decision_price")):
                tracked = lifecycle["first_qualified_line"]
                if tracked:
                    decision_line, decision_price = _line_and_price(_text(row.get("market")).upper(), _text(row.get("side")), tracked)
                    ledger.at[index, "decision_line"] = decision_line
                    ledger.at[index, "decision_price"] = decision_price
                    ledger.at[index, "decision_line_source"] = "first_qualified_capture"
                    ledger.at[index, "decision_quote_observed_at"] = lifecycle["entry_observed_at"]
                elif _truthy(row.get("favorite_qualified")):
                    ledger.at[index, "decision_line"] = row.get("final_pregame_line", "")
                    ledger.at[index, "decision_price"] = row.get("final_pregame_price", "")
                    ledger.at[index, "decision_line_source"] = "legacy_final_pregame_fallback"
            for field in ("closing_line", "closing_price"):
                fallback = row.get(field.replace("closing_", "final_pregame_"), "")
                if not _text(row.get(f"{field}_source")):
                    ledger.at[index, f"{field}_source"] = _closing_source(row, field, fallback)
                if not _text(row.get(field)):
                    ledger.at[index, field] = fallback
            if not _text(row.get("price_band")):
                ledger.at[index, "price_band"] = _price_band(row.get("final_pregame_price")) if _text(row.get("market")).upper() == "MONEYLINE" else "not_applicable"
            if not _text(row.get("spread_band")):
                ledger.at[index, "spread_band"] = _spread_band(row.get("final_pregame_line"), _text(row.get("market")).upper())
            if not _text(row.get("bets_pct_band")):
                ledger.at[index, "bets_pct_band"] = _percent_band(row.get("bets_pct"))
            if not _text(row.get("money_pct_band")):
                ledger.at[index, "money_pct_band"] = _percent_band(row.get("money_pct"))
            if not _text(row.get("movement_magnitude_band")):
                ledger.at[index, "movement_magnitude_band"] = _movement_band(row.get("movement_magnitude"))
            if not _text(row.get("favorite_cross_market_state")):
                ledger.at[index, "favorite_cross_market_state"] = "unavailable"
            if not _text(row.get("first_qualified_timing_band")):
                ledger.at[index, "first_qualified_timing_band"] = _timing_band(row.get("minutes_first_qualified_before_start"))
            if not _text(row.get("final_state_timing_band")):
                ledger.at[index, "final_state_timing_band"] = _timing_band(row.get("minutes_final_state_before_start"))
    before_classification = ledger.reindex(columns=CLASSIFICATION_COLUMNS).copy()
    ledger = ledger.reindex(columns=LEDGER_COLUMNS)
    if attach_results:
        score_frames = [_read_csv(data_dir / "final_scores_history.csv")]
        score_frames.extend(_scores_from_frozen(frame) for frame in source_frames)
        score_frames.append(_refresh_ufc_result_history(ledger, data_dir))
        if not ledger.empty and ledger["freeze_method"].eq(ufc.HANDOFF).any():
            bouts, _ = ufc.load_context(data_dir)
            score_frames.append(ufc.result_rows(ledger, bouts))
        scores = pd.concat([frame for frame in score_frames if not frame.empty], ignore_index=True, sort=False) if any(not frame.empty for frame in score_frames) else pd.DataFrame()
        mlb_cutover = mlb_cutover_at(data_dir)
        if mlb_cutover and not scores.empty:
            # Legacy history can have an unverified score for a pending ID.
            # Prefer a fixture-proven final for MLB only. Existing grade rows
            # and the history CSV are left intact.
            verified = scores.get("score_source", pd.Series("", index=scores.index)).eq(MLB_SCORE_SOURCE)
            mlb_ids = set(ledger.loc[ledger["sport"].eq("mlb"), "event_id"])
            superseded = ~verified & scores["game_id"].isin(mlb_ids & set(scores.loc[verified, "game_id"]))
            scores = scores.loc[~superseded]
        ledger = _attach_results(ledger, scores, mlb_cutover=mlb_cutover)
    else:
        for index, row in ledger.iterrows():
            _attach_clv(ledger, index, row)
    # Outcome attachment may never mutate the frozen classification fields.
    pd.testing.assert_frame_equal(before_classification.reset_index(drop=True), ledger.reindex(columns=CLASSIFICATION_COLUMNS).reset_index(drop=True))
    _atomic_csv(ledger.reindex(columns=LEDGER_COLUMNS), ledger_path)
    _exports(ledger, data_dir)
    _update_ncaaf_shadow_results(ledger, data_dir)
    graded = ledger[ledger["grade"].isin(["W", "L", "Push"])]
    result = {
        "rows": int(len(ledger)), "new_rows": int(len(new_records)), "graded": int(len(graded)),
        "supported": summary(graded),
        "favorites": summary(graded[graded["favorite_qualified"].eq("yes")]),
    }
    favorite_rows = graded[graded["favorite_qualified"].eq("yes")]
    favorite_sport = favorite_rows.get("sport", pd.Series("", index=favorite_rows.index, dtype=str)).astype(str).str.strip().str.lower()
    sport_records = []
    for sport, label in FAVORITE_SPORTS:
        sport_records.append({"sport": sport, "label": label, **summary(favorite_rows[favorite_sport.eq(sport)])})
    sport_records.sort(key=lambda record: (
        record["win_rate_excluding_pushes"] == "",
        -float(record["win_rate_excluding_pushes"] or 0),
        record["label"],
    ))
    new_path_records = {}
    for pathway, version in CFB_SPREAD_VERSIONS.items():
        selected = favorite_rows.loc[
            favorite_rows["favorite_pathway"].eq(pathway) & favorite_rows["favorite_rule_version"].eq(version)
        ]
        if not selected.empty:
            net = pd.to_numeric(selected["net_profit_units"], errors="coerce")
            units = pd.to_numeric(selected["stake_units"], errors="coerce")
            stakes = float(units.loc[net.notna()].sum())
            new_path_records[pathway] = {
                **summary(selected), "net_units": round(float(net.sum()), 6),
                "priced_stakes": stakes, "roi": round(float(net.sum()) / stakes, 6) if stakes else None,
            }
    nhl_path_records = {
        pathway: summary(favorite_rows.loc[favorite_rows["favorite_rule_version"].eq(version)])
        for pathway, version in OFFICIAL_VERSIONS.items()
    }
    first_qualified = pd.to_datetime(
        favorite_rows["favorite_first_qualified_at"], utc=True, format="mixed", errors="coerce",
    )
    _atomic_json({
        **result["favorites"],
        **({"cfb_spread_pathways": new_path_records} if new_path_records else {}),
        **({"nhl_replacement_pathways": nhl_path_records}
           if ledger["favorite_rule_version"].isin(OFFICIAL_VERSIONS.values()).any() else {}),
        **({"nba_public_neutral_moneyline": _nba_public_performance(ledger, data_dir)}
           if (data_dir / "nba_public_neutral_activated_at").exists() or ledger["favorite_rule_version"].eq(NBA_PUBLIC_VERSION).any() else {}),
        **({"mlb_static_moneyline": _version_performance(ledger, MLB_STATIC_VERSION, data_dir / "mlb_static_activated_at")}
           if (data_dir / "mlb_static_activated_at").exists() or ledger["favorite_rule_version"].eq(MLB_STATIC_VERSION).any() else {}),
        "sports": sport_records,
        "logic_change_records": {
            sport: {
                **summary(favorite_rows.loc[favorite_sport.eq(sport) & first_qualified.ge(cutover)]),
                "since": cutover.tz_convert("America/New_York").date().isoformat(),
            }
            for sport, cutover in FAVORITE_LOGIC_CUTOVERS.items()
        },
        "games": favorite_rows.sort_values(
            ["scheduled_start", "event_id"], ascending=False, kind="mergesort",
        )[[
            "sport", "game", "scheduled_start", "market", "side", "open_line", "open_price",
            "decision_line", "decision_price", "decision_line_source", "decision_quote_observed_at",
            "closing_line", "closing_price", "closing_line_source", "closing_price_source",
            "closing_line_observed_at", "closing_price_observed_at", "final_score", "grade",
            "bets_pct", "money_pct", "favorite_origin",
        ]].fillna("").to_dict("records"),
        "tracking_start_date": FAVORITE_TRACKING_START_DATE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, data_dir / "favorite_performance.json")
    return result


def _nba_public_performance(ledger: pd.DataFrame, data_dir: Path) -> dict:
    return _version_performance(ledger, NBA_PUBLIC_VERSION, data_dir / "nba_public_neutral_activated_at")


def _version_performance(ledger: pd.DataFrame, version: str, marker: Path) -> dict:
    rows = ledger.loc[ledger["favorite_rule_version"].eq(version) & ledger["favorite_qualified"].eq("yes")]
    result = _cohort_row(rows, "favorite_rule_version", version, "accepted")
    return {**summary(rows), **{key: result[key] for key in ("stake_units", "net_profit_units", "roi")},
            "activated_at": marker.read_text(encoding="utf8").strip() if marker.exists() else "",
            "first_qualified_at": min((v for v in rows["favorite_first_qualified_at"] if v), default="")}


def summary(rows: pd.DataFrame) -> dict:
    wins = int(rows.get("grade", pd.Series(dtype=str)).eq("W").sum())
    losses = int(rows.get("grade", pd.Series(dtype=str)).eq("L").sum())
    pushes = int(rows.get("grade", pd.Series(dtype=str)).eq("Push").sum())
    return {
        "wins": wins, "losses": losses, "pushes": pushes, "total_graded": wins + losses + pushes,
        "win_rate_excluding_pushes": round(wins / (wins + losses), 6) if wins + losses else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the admin performance CSV ledger")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--frozen-source", action="append", default=[], help="Archived live_recent CSV; repeatable")
    parser.add_argument("--freeze-only", action="store_true", help="Ingest new frozen rows without outcome grading")
    args = parser.parse_args()
    paths = [Path(value) for value in args.frozen_source] or None
    print(json.dumps(update_performance_ledger(args.data_dir, paths, attach_results=not args.freeze_only), indent=2))
