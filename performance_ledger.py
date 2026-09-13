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

from anomaly_action_results import _grade_action


SUPPORTED_SIDE_VALID_FROM = pd.Timestamp("2026-09-07T17:57:58Z")
FAVORITE_VALID_FROM = pd.Timestamp("2026-09-07T22:11:12Z")
FAVORITE_TRACKING_START_DATE = "2026-09-07"
FAVORITE_KPI_EXCLUSIONS = frozenset({
    ("34603681", "SPREAD"),  # Jacksonville State @ Ohio, visibility invalidated
})
LEDGER_COLUMNS = [
    "ledger_id", "event_id", "game", "sport", "scheduled_start", "market", "side",
    "open_line", "open_price", "decision_line", "decision_price", "decision_line_source",
    "final_pregame_line", "final_pregame_price",
    "line_move_from_open", "market_read", "market_read_detail", "market_path",
    "observation_count", "line_direction_changes", "return_toward_open",
    "active_worsening_reversal", "whipsaw_state", "whipsaw_ever",
    "supported_side", "favorite_qualified", "favorite_rule_version",
    "favorite_originally_qualified", "favorite_late_invalidated",
    "favorite_late_invalidated_at", "favorite_late_invalidated_reason",
    "favorite_pathway", "candidate_decision", "candidate_decision_reason",
    "price_band", "spread_band",
    "favorite_first_qualified_at", "favorite_final_qualified_at",
    "favorite_first_fell_off_at", "favorite_last_fell_off_at",
    "favorite_fell_off_before_kickoff", "qualification_episode_count",
    "requalification_count", "falloff_count", "qualifying_capture_count",
    "minutes_first_qualified_before_start", "minutes_final_state_before_start",
    "first_qualified_timing_band", "final_state_timing_band",
    "final_pregame_state_at_utc", "final_pregame_frozen_at",
    "final_score", "market_result", "grade", "stake_units", "net_profit_units", "roi",
    "closing_line", "closing_price", "clv", "clv_method", "clv_available",
    "score_provider", "score_provider_event_id", "score_resolved_at",
    "freeze_source", "freeze_method", "source_snapshot_id", "source_hash", "audit_status",
]
CLASSIFICATION_COLUMNS = LEDGER_COLUMNS[:LEDGER_COLUMNS.index("final_score")] + [
    "freeze_source", "freeze_method", "source_snapshot_id", "source_hash",
]
RESULT_COLUMNS = [
    "final_score", "market_result", "grade", "stake_units", "net_profit_units", "roi",
    "closing_line", "closing_price", "clv", "clv_method", "clv_available",
    "score_provider", "score_provider_event_id", "score_resolved_at", "audit_status",
]
EXPORT_FILES = {
    "combined": "performance_combined.csv",
    "supported": "performance_supported_sides.csv",
    "favorites": "performance_favorites.csv",
    "candidate_audit": "favorite_candidate_audit.csv",
    "cohort_kpis": "favorite_cohort_kpis.csv",
}

COHORT_DIMENSIONS = [
    "sport", "market", "favorite_pathway", "price_band", "spread_band",
    "first_qualified_timing_band", "final_state_timing_band", "whipsaw_ever",
    "favorite_fell_off_before_kickoff",
]
COHORT_KPI_COLUMNS = [
    "dimension", "segment", "candidate_decision", "candidates", "graded",
    "wins", "losses", "pushes", "win_rate_excluding_pushes", "stake_units",
    "net_profit_units", "roi", "clv_available", "clv_coverage", "average_clv",
    "whipsaw_candidates", "whipsaw_rate", "fell_off_candidates", "falloff_rate",
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
    return parsed


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


def _tracking_lifecycle(history: pd.DataFrame, row: pd.Series, kickoff: pd.Timestamp) -> dict[str, str]:
    empty = {
        "first_qualified": "", "last_qualified": "", "first_fell_off": "",
        "last_fell_off": "", "episodes": "0", "requalifications": "0",
        "falloffs": "0", "qualifying_captures": "0", "whipsaw_ever": "no",
        "pathway": "", "first_qualified_line": "",
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
    scoped["_at"] = pd.to_datetime(scoped.get("recorded_at", ""), errors="coerce", utc=True)
    scoped = scoped[scoped["_at"].notna()]
    if pd.notna(kickoff):
        scoped = scoped[scoped["_at"] <= kickoff]
    scoped = scoped.sort_values("_at", kind="mergesort")
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
    first_lines = scoped.loc[qualified, "first_qualified_line"] if "first_qualified_line" in scoped else pd.Series(dtype=str)
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
        "first_qualified_line": _text(first_lines.iloc[0]) if not first_lines.empty else "",
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
    ordered = frozen.copy()
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
    ordered = ordered.sort_values(["_state_at", "_frozen"], kind="mergesort").drop_duplicates(key_columns, keep="last")
    for _, row in ordered.iterrows():
        kickoff = pd.to_datetime(row.get("kickoff_iso", ""), errors="coerce", utc=True)
        if pd.isna(kickoff) or kickoff < SUPPORTED_SIDE_VALID_FROM:
            continue
        supported = _text(row.get("supported_side"))
        if not supported:
            continue
        side_record = _side_record(row, supported)
        if not side_record:
            # A direction without one exact source-side match is not auditable.
            continue
        market = _text(row.get("market_display")).upper()
        line, price = _line_and_price(market, supported, side_record.get("current_line"))
        open_line, open_price = _line_and_price(market, supported, side_record.get("open_line"))
        lifecycle = _tracking_lifecycle(history, row, kickoff)
        first_qualified, fell_off = _lifecycle(history, row, kickoff)
        late_invalidated = _truthy(row.get("favorite_late_invalidated"))
        originally_qualified = _truthy(row.get("favorite_originally_qualified")) or late_invalidated
        favorite = (
            kickoff >= FAVORITE_VALID_FROM
            and _text(row.get("favorite_rule_version")) == "red_fox_favorite_v2"
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
        if not tracked_decision and favorite:
            decision_line, decision_price = line, price
        decision_source = "first_qualified_capture" if tracked_decision else ("final_pregame_fallback" if favorite else "")
        movement_start = _number(open_price if market == "MONEYLINE" else open_line)
        movement_end = _number(price if market == "MONEYLINE" else line)
        line_move = "" if movement_start is None or movement_end is None else str(round(movement_end - movement_start, 4))
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
            "market": market, "side": supported, "open_line": open_line, "open_price": open_price,
            "decision_line": decision_line, "decision_price": decision_price,
            "decision_line_source": decision_source,
            "final_pregame_line": line,
            "final_pregame_price": price,
            "line_move_from_open": line_move,
            "market_read": _text(side_record.get("reaction")),
            "market_read_detail": _text(side_record.get("anomaly_chips")) or _text(side_record.get("reaction")),
            "market_path": _text(side_record.get("path")),
            "observation_count": _text(side_record.get("observation_count")),
            "line_direction_changes": _text(side_record.get("line_dir_changes")),
            "return_toward_open": "yes" if _truthy(side_record.get("return_toward_open")) else "no",
            "active_worsening_reversal": "yes" if _truthy(side_record.get("active_worsening_reversal")) else "no",
            "whipsaw_state": whipsaw_state, "whipsaw_ever": whipsaw_ever,
            "supported_side": supported, "favorite_qualified": "yes" if favorite else "no",
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
            "final_score": "", "market_result": "", "grade": "",
            "stake_units": "", "net_profit_units": "", "roi": "",
            "closing_line": _text(row.get("closing_line")) or line,
            "closing_price": _text(row.get("closing_price")) or price,
            "clv": _text(row.get("clv")), "clv_method": "retained_source" if _text(row.get("clv")) else "",
            "clv_available": "yes" if _text(row.get("clv")) else "no",
            "score_provider": "", "score_provider_event_id": "",
            "score_resolved_at": "", "freeze_source": source_name,
            "freeze_method": _text(row.get("freeze_method")) or "audited_latest_state_at_or_before_start",
            "source_snapshot_id": _text(row.get("favorite_snapshot_id")),
            "source_hash": hashlib.sha256(source_payload.encode("utf-8")).hexdigest(),
            "audit_status": "awaiting_final_score",
        })
    return records


def _score_map(scores: pd.DataFrame) -> dict[str, pd.Series]:
    if scores.empty or "game_id" not in scores:
        return {}
    return {str(row.get("game_id", "")): row for _, row in scores.drop_duplicates("game_id", keep="last").iterrows()}


def _scores_from_frozen(frozen: pd.DataFrame) -> pd.DataFrame:
    """Use already-retained final live scores; this performs no API call."""
    if frozen.empty or "score_state" not in frozen:
        return pd.DataFrame()
    final = frozen[frozen["score_state"].astype(str).str.lower().eq("post")].copy()
    records = []
    for _, row in final.drop_duplicates("game_id", keep="last").iterrows():
        game = _text(row.get("game"))
        if " @ " not in game:
            continue
        away, home = game.split(" @ ", 1)
        records.append({
            "game_id": _text(row.get("game_id")), "team1": away,
            "team1_score": _text(row.get("score_away")), "team2": home,
            "team2_score": _text(row.get("score_home")),
            "score_provider": _text(row.get("score_provider")) or "retained_live_recent",
            "score_provider_event_id": _text(row.get("score_provider_event_id")),
            "resolved_at_utc": _text(row.get("score_completed_at_utc")) or _text(row.get("score_updated_at_utc")),
        })
    return pd.DataFrame(records)


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
    if _text(row.get("clv")):
        result.at[index, "clv_available"] = "yes"
        result.at[index, "clv_method"] = _text(row.get("clv_method")) or "retained_source"
        return
    market = _text(row.get("market")).upper()
    if market == "MONEYLINE":
        decision = _implied_probability(row.get("decision_price"))
        closing = _implied_probability(row.get("closing_price"))
        if decision is not None and closing is not None:
            result.at[index, "clv"] = str(round((closing - decision) * 100.0, 4))
            result.at[index, "clv_method"] = "implied_probability_points"
            result.at[index, "clv_available"] = "yes"
            return
    elif market == "SPREAD":
        decision = _number(row.get("decision_line"))
        closing = _number(row.get("closing_line"))
        if decision is not None and closing is not None:
            result.at[index, "clv"] = str(round(decision - closing, 4))
            result.at[index, "clv_method"] = "side_line_points"
            result.at[index, "clv_available"] = "yes"
            return
    result.at[index, "clv_available"] = "no"


def _attach_results(ledger: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    result = ledger.copy()
    for index, row in result.iterrows():
        _attach_clv(result, index, row)
        grade = _text(row.get("grade"))
        win_profit = _american_profit(row.get("decision_price"))
        if grade in {"W", "L", "Push"} and win_profit is not None and not _text(row.get("stake_units")):
            net = win_profit if grade == "W" else (-1.0 if grade == "L" else 0.0)
            result.at[index, "stake_units"] = "1"
            result.at[index, "net_profit_units"] = f"{round(net, 6):g}"
            result.at[index, "roi"] = f"{round(net, 6):g}"
    by_event = _score_map(scores)
    unresolved = ~result.get("grade", pd.Series("", index=result.index)).isin(["W", "L", "Push"])
    for index, row in result[unresolved].iterrows():
        score = by_event.get(_text(row.get("event_id")))
        if score is None:
            if not _text(row.get("grade")):
                result.at[index, "audit_status"] = "awaiting_final_score"
            continue
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
        result.at[index, "final_score"] = f"{team1} {score1} - {team2} {score2}"
        result.at[index, "market_result"] = f"{_text(row.get('side'))}: {grade.title()}"
        result.at[index, "grade"] = {"WIN": "W", "LOSS": "L", "PUSH": "Push"}[grade]
        win_profit = _american_profit(row.get("decision_price"))
        if win_profit is not None:
            result.at[index, "stake_units"] = "1"
            net = win_profit if grade == "WIN" else (-1.0 if grade == "LOSS" else 0.0)
            result.at[index, "net_profit_units"] = f"{round(net, 6):g}"
            result.at[index, "roi"] = f"{round(net, 6):g}"
        result.at[index, "score_provider"] = _text(score.get("score_provider")) or "espn_final_scores_history"
        result.at[index, "score_provider_event_id"] = _text(score.get("score_provider_event_id"))
        result.at[index, "score_resolved_at"] = _text(score.get("resolved_at_utc"))
        result.at[index, "audit_status"] = "graded"
    return result


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
    clv = pd.to_numeric(rows.get("clv", pd.Series(dtype=str)), errors="coerce")
    clv_count = int(clv.notna().sum())
    whipsaw = rows.get("whipsaw_ever", pd.Series("no", index=rows.index)).astype(str).eq("yes")
    fell_off = rows.get("favorite_fell_off_before_kickoff", pd.Series("no", index=rows.index)).astype(str).eq("yes")
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
        frame = _read_csv(Path(source))
        source_frames.append(frame)
        candidates.extend(_frozen_records(frame, history, str(Path(source))))
    existing_ids = set(ledger.get("ledger_id", pd.Series(dtype=str)).astype(str))
    new_records = [record for record in candidates if record["ledger_id"] not in existing_ids]
    if new_records:
        ledger = pd.concat([ledger, pd.DataFrame(new_records)], ignore_index=True, sort=False)
    if not ledger.empty:
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
        for index, row in ledger.iterrows():
            if not _text(row.get("decision_line")) and not _text(row.get("decision_price")):
                lifecycle_row = pd.Series({
                    "sport": row.get("sport", ""), "game_id": row.get("event_id", ""),
                    "market_display": row.get("market", ""),
                })
                lifecycle = _tracking_lifecycle(
                    history, lifecycle_row,
                    pd.to_datetime(row.get("scheduled_start", ""), errors="coerce", utc=True),
                )
                tracked = lifecycle["first_qualified_line"]
                if tracked:
                    decision_line, decision_price = _line_and_price(_text(row.get("market")).upper(), _text(row.get("side")), tracked)
                    ledger.at[index, "decision_line"] = decision_line
                    ledger.at[index, "decision_price"] = decision_price
                    ledger.at[index, "decision_line_source"] = "first_qualified_capture"
                elif _truthy(row.get("favorite_qualified")):
                    ledger.at[index, "decision_line"] = row.get("final_pregame_line", "")
                    ledger.at[index, "decision_price"] = row.get("final_pregame_price", "")
                    ledger.at[index, "decision_line_source"] = "legacy_final_pregame_fallback"
            if not _text(row.get("closing_line")):
                ledger.at[index, "closing_line"] = row.get("final_pregame_line", "")
            if not _text(row.get("closing_price")):
                ledger.at[index, "closing_price"] = row.get("final_pregame_price", "")
            if not _text(row.get("price_band")):
                ledger.at[index, "price_band"] = _price_band(row.get("final_pregame_price")) if _text(row.get("market")).upper() == "MONEYLINE" else "not_applicable"
            if not _text(row.get("spread_band")):
                ledger.at[index, "spread_band"] = _spread_band(row.get("final_pregame_line"), _text(row.get("market")).upper())
            if not _text(row.get("first_qualified_timing_band")):
                ledger.at[index, "first_qualified_timing_band"] = _timing_band(row.get("minutes_first_qualified_before_start"))
            if not _text(row.get("final_state_timing_band")):
                ledger.at[index, "final_state_timing_band"] = _timing_band(row.get("minutes_final_state_before_start"))
    before_classification = ledger.reindex(columns=CLASSIFICATION_COLUMNS).copy()
    ledger = ledger.reindex(columns=LEDGER_COLUMNS)
    if attach_results:
        score_frames = [_read_csv(data_dir / "final_scores_history.csv")]
        score_frames.extend(_scores_from_frozen(frame) for frame in source_frames)
        scores = pd.concat([frame for frame in score_frames if not frame.empty], ignore_index=True, sort=False) if any(not frame.empty for frame in score_frames) else pd.DataFrame()
        ledger = _attach_results(ledger, scores)
    # Outcome attachment may never mutate the frozen classification fields.
    pd.testing.assert_frame_equal(before_classification.reset_index(drop=True), ledger.reindex(columns=CLASSIFICATION_COLUMNS).reset_index(drop=True))
    _atomic_csv(ledger.reindex(columns=LEDGER_COLUMNS), ledger_path)
    _exports(ledger, data_dir)
    graded = ledger[ledger["grade"].isin(["W", "L", "Push"])]
    result = {
        "rows": int(len(ledger)), "new_rows": int(len(new_records)), "graded": int(len(graded)),
        "supported": summary(graded),
        "favorites": summary(graded[graded["favorite_qualified"].eq("yes")]),
    }
    _atomic_json({
        **result["favorites"],
        "tracking_start_date": FAVORITE_TRACKING_START_DATE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, data_dir / "favorite_performance.json")
    return result


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
