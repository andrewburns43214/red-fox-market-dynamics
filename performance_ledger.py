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
from pathlib import Path
from uuid import uuid4

import pandas as pd

from anomaly_action_results import _grade_action


SUPPORTED_SIDE_VALID_FROM = pd.Timestamp("2026-09-07T17:57:58Z")
FAVORITE_VALID_FROM = pd.Timestamp("2026-09-07T22:11:12Z")
LEDGER_COLUMNS = [
    "ledger_id", "event_id", "game", "sport", "scheduled_start", "market", "side",
    "final_pregame_line", "final_pregame_price", "market_read", "market_read_detail",
    "supported_side", "favorite_qualified", "favorite_rule_version",
    "favorite_first_qualified_at", "favorite_final_qualified_at",
    "favorite_fell_off_before_kickoff", "final_pregame_state_at_utc", "final_pregame_frozen_at",
    "final_score", "market_result", "grade", "closing_line", "closing_price", "clv",
    "score_provider", "score_provider_event_id", "score_resolved_at",
    "freeze_source", "freeze_method", "source_snapshot_id", "source_hash", "audit_status",
]
CLASSIFICATION_COLUMNS = LEDGER_COLUMNS[:19] + [
    "freeze_source", "freeze_method", "source_snapshot_id", "source_hash",
]
RESULT_COLUMNS = [
    "final_score", "market_result", "grade", "closing_line", "closing_price", "clv",
    "score_provider", "score_provider_event_id", "score_resolved_at", "audit_status",
]
EXPORT_FILES = {
    "combined": "performance_combined.csv",
    "supported": "performance_supported_sides.csv",
    "favorites": "performance_favorites.csv",
}


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
        first_qualified, fell_off = _lifecycle(history, row, kickoff)
        favorite = (
            kickoff >= FAVORITE_VALID_FROM
            and _text(row.get("favorite_rule_version")) == "red_fox_favorite_v2"
            and _truthy(row.get("red_fox_favorite"))
            and _identity(row.get("favorite_side")) == _identity(supported)
        )
        frozen_at = _text(row.get("frozen_at_utc"))
        state_at = row["_state_at"].isoformat()
        source_payload = "|".join([
            _text(row.get("sport")), _text(row.get("game_id")), market, supported,
            _text(side_record.get("current_line")), frozen_at, _text(row.get("favorite_snapshot_id")),
        ])
        ledger_id = hashlib.sha256("|".join(source_payload.split("|")[:3]).encode("utf-8")).hexdigest()[:20]
        records.append({
            "ledger_id": ledger_id,
            "event_id": _text(row.get("game_id")), "game": _text(row.get("game")),
            "sport": _text(row.get("sport")), "scheduled_start": _text(row.get("kickoff_iso")),
            "market": market, "side": supported, "final_pregame_line": line,
            "final_pregame_price": price,
            "market_read": _text(side_record.get("reaction")),
            "market_read_detail": _text(side_record.get("anomaly_chips")) or _text(side_record.get("reaction")),
            "supported_side": supported, "favorite_qualified": "yes" if favorite else "no",
            "favorite_rule_version": _text(row.get("favorite_rule_version")),
            "favorite_first_qualified_at": first_qualified,
            "favorite_final_qualified_at": _text(row.get("favorite_final_qualified_at")) if favorite else "",
            "favorite_fell_off_before_kickoff": fell_off,
            "final_pregame_state_at_utc": state_at,
            "final_pregame_frozen_at": frozen_at,
            "final_score": "", "market_result": "", "grade": "",
            "closing_line": _text(row.get("closing_line")), "closing_price": _text(row.get("closing_price")),
            "clv": _text(row.get("clv")), "score_provider": "", "score_provider_event_id": "",
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


def _attach_results(ledger: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    result = ledger.copy()
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
        result.at[index, "score_provider"] = _text(score.get("score_provider")) or "espn_final_scores_history"
        result.at[index, "score_provider_event_id"] = _text(score.get("score_provider_event_id"))
        result.at[index, "score_resolved_at"] = _text(score.get("resolved_at_utc"))
        result.at[index, "audit_status"] = "graded"
    return result


def _exports(ledger: pd.DataFrame, data_dir: Path) -> None:
    combined = ledger.reindex(columns=LEDGER_COLUMNS).sort_values(["scheduled_start", "event_id", "market"], kind="mergesort")
    _atomic_csv(combined, data_dir / EXPORT_FILES["combined"])
    _atomic_csv(combined, data_dir / EXPORT_FILES["supported"])
    favorites = combined[combined["favorite_qualified"].eq("yes")].copy()
    _atomic_csv(favorites, data_dir / EXPORT_FILES["favorites"])


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
    return {
        "rows": int(len(ledger)), "new_rows": int(len(new_records)), "graded": int(len(graded)),
        "supported": summary(graded),
        "favorites": summary(graded[graded["favorite_qualified"].eq("yes")]),
    }


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
