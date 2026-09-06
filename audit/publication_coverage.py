"""Offline publication reconciliation. Never called by the production runner.

Reads immutable copies only; does not score, publish, fetch ESPN, or modify gates.
An inventory is one row per discovered (sport, game_id, market_display), including
markets whose split rows could not be parsed. Missing evidence is not a zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from anomaly_board import _build_history_points, _parse_snapshot_value
from main import infer_market_type, normalize_side_key
from refresh_anomaly_board import (
    complete_public_market_rows,
    filter_fresh_market_rows,
    filter_publication_eligible_markets,
    latest_synchronized_market_rows,
)

KEYS = ["sport", "game_id", "market_display"]
SPORTS = {"nfl", "ncaaf", "nba", "ncaab", "mlb", "nhl", "ufc"}
MARKETS = {"MONEYLINE", "SPREAD", "TOTAL"}
HEADER_MARKETS = {"RUN LINE": "SPREAD", "PUCK LINE": "SPREAD"}


def season_enabled(sport, now):
    """Audit the runner's existing calendar switches; do not widen any season."""
    mmdd = int(now.strftime("%m%d"))
    bounds = dict(nfl=(801, 225), ncaaf=(801, 201), mlb=(301, 1115),
                  nba=(1001, 715), nhl=(920, 715), ncaab=(1001, 415))
    if sport == "ufc":
        return True
    if sport not in bounds:
        return False
    start, end = bounds[sport]
    return start <= mmdd <= end if start <= end else mmdd >= start or mmdd <= end


def utc(value):
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


from dk_discovery import inventory_html


def prepare(snapshots):
    work = snapshots.copy().fillna("")
    for col in [*KEYS, "timestamp", "side", "current_line", "open_line", "bets_pct", "money_pct", "dk_start_iso"]:
        if col not in work:
            work[col] = ""
    # Deliberately use the publisher's inference, including its limitations.
    work["market_display"] = work.apply(
        lambda r: infer_market_type(r.side, r.current_line), axis=1) if len(work) else pd.Series(dtype=str)
    work["side_key"] = work.apply(
        lambda r: normalize_side_key(r.sport, r.market_display, r.side), axis=1) if len(work) else pd.Series(dtype=str)
    work["timestamp"] = pd.to_datetime(work.timestamp, utc=True, errors="coerce")
    return work


def keyset(frame):
    return set(frame[KEYS].itertuples(index=False, name=None)) if len(frame) else set()


def captured_inventory(snapshots):
    """Lower-bound inventory; cannot establish source discovery completeness."""
    work = prepare(snapshots).sort_values("timestamp", na_position="first")
    # Recover source market identity for the documented wide-odds parser defect.
    # This is inventory only: prepare/reconciliation still run production inference.
    wide_odds = work.market_display.eq("") & work.current_line.str.contains(r"@\s*[+-]\d{5,}\s*$", regex=True)
    work.loc[wide_odds, "market_display"] = "MONEYLINE"
    out = work.drop_duplicates(KEYS, keep="last")[KEYS + ["game", "dk_start_iso"]].copy()
    out["discovery_basis"] = "CAPTURED_ONLY_NOT_SOURCE_CENSUS"
    out["league_identified"] = False  # assigned sport is not independent league evidence
    out["identity_verified"] = out.game_id.ne("")
    return out


def reconcile(inventory, snapshots, board, now, max_age_minutes=10):
    """Account for every inventory key, reusing actual publication gate functions.

    None snapshots/board means unavailable evidence, never an empty source/export.
    Caller must supply copies aligned to one publication run to interpret gaps.
    """
    now = utc(now)
    inv = inventory.copy().fillna("")
    if inv.empty:
        return pd.DataFrame(columns=[*KEYS, "status", "in_window_target", "publication_eligible", "observed_published"])
    if inv.duplicated(KEYS).any():
        raise ValueError("Inventory must contain one record per discovered market")
    work = prepare(snapshots if snapshots is not None else pd.DataFrame())
    # Future capture timestamps cannot be used in a historical replay.
    work = work[work.timestamp.isna() | (work.timestamp <= now)].copy()
    supported = work[work.market_display.isin(MARKETS)]
    newest = supported.timestamp.max()
    active = supported[supported.timestamp >= newest - pd.Timedelta(hours=2)].copy()
    paired = latest_synchronized_market_rows(active)
    complete = complete_public_market_rows(paired)
    fresh = filter_fresh_market_rows(complete, now=now, max_age_minutes=max_age_minutes)
    capture_kickoff = pd.to_datetime(fresh.dk_start_iso, utc=True, errors="coerce")
    pregame_rows = fresh[capture_kickoff.notna() & (capture_kickoff > now - pd.Timedelta(minutes=5))]
    window_rows = filter_publication_eligible_markets(pregame_rows, now=now)
    sets = {name: keyset(df) for name, df in [("captured", work), ("active", active),
            ("paired", paired), ("complete", complete), ("fresh", fresh),
            ("pregame", pregame_rows), ("window", window_rows)]}
    paired_groups = dict(tuple(paired.groupby(KEYS))) if len(paired) else {}
    history_groups = dict(tuple(supported.groupby(KEYS))) if len(supported) else {}
    published = keyset(board) if board is not None else set()
    results = []
    for record in inv.to_dict("records"):
        key = tuple(record[k] for k in KEYS)
        captured = key in sets["captured"]
        history = history_groups.get(key, supported.iloc[:0])
        pair = paired_groups.get(key, paired.iloc[:0])
        kickoff_text = record.get("dk_start_iso", "")
        if not kickoff_text and len(history):
            kickoff_text = history.sort_values("timestamp").iloc[-1].dk_start_iso
        kickoff = pd.to_datetime(kickoff_text, utc=True, errors="coerce")
        window = "UNKNOWN"
        if pd.notna(kickoff):
            probe = pd.DataFrame([dict(sport=key[0], dk_start_iso=kickoff.isoformat())])
            window = "INSIDE_PUBLICATION_WINDOW" if len(filter_publication_eligible_markets(probe, now=now)) else "OUTSIDE_PUBLICATION_WINDOW"
        supported_market = key[0] in SPORTS and key[2] in MARKETS and not (key[0] == "ufc" and key[2] != "MONEYLINE")
        enabled = season_enabled(key[0], now)
        pregame = pd.notna(kickoff) and kickoff > now - pd.Timedelta(minutes=5)
        unverified_league = record.get("discovery_basis") == "RAW_DK_HEADER" and str(record.get("league_identified", False)).lower() != "true"
        target = bool(supported_market and enabled and not unverified_league and window == "INSIDE_PUBLICATION_WINDOW" and pregame)
        reason = ""
        detail = ""
        history_counts = {}
        if unverified_league:
            reason = "SPORT_LEAGUE_NOT_VERIFIED"
            window = "UNKNOWN"
        elif window == "OUTSIDE_PUBLICATION_WINDOW":
            reason = window
        elif not supported_market:
            reason = "UNSUPPORTED_SPORT_OR_MARKET"
        elif str(record.get("identity_verified", True)).lower() == "false" or not key[1]:
            reason = "UNRESOLVED_EVENT_IDENTITY"
        elif pd.isna(kickoff):
            reason = "MISSING_OR_INVALID_KICKOFF"
        elif not pregame:
            reason = "KICKOFF_EXPIRED"
        elif not enabled:
            reason = "SPORT_DISABLED_BY_SEASON"
        elif record.get("capture_exclusion_reason"):
            reason = str(record["capture_exclusion_reason"])
        elif snapshots is None:
            reason = "CAPTURE_EVIDENCE_UNAVAILABLE"
        elif not captured:
            reason = "NOT_CAPTURED_OR_MARKET_NOT_NORMALIZED"
        elif key not in sets["active"]:
            reason = "NO_CAPTURE_IN_TWO_HOUR_WORKING_SET"
        elif key not in sets["paired"]:
            reason = "NO_SYNCHRONIZED_SIDE_PAIR"
            rejected = work[work.sport.eq(key[0]) & work.game_id.eq(key[1]) & work.market_display.eq("")]
            if key[2] == "MONEYLINE" and rejected.current_line.str.contains(r"@\s*[+-]\d{5,}\s*$", regex=True).any():
                reason = "MONEYLINE_ODDS_WIDTH_NORMALIZATION_FAILURE"
                detail = json.dumps(sorted(set(rejected.current_line)))
        elif key not in sets["complete"]:
            reason = "INCOMPLETE_MARKET_FIELDS_OR_SIDE_COUNT"
            required = ["side_key", "side", "bets_pct", "money_pct", "open_line", "current_line", "dk_start_iso"]
            detail = json.dumps({"side_count": int(pair.side_key.nunique()), "missing_fields":
                                 [c for c in required if pair[c].fillna("").astype(str).str.strip().eq("").any()]})
        elif key not in sets["fresh"]:
            reason = "STALE_CAPTURE"
        elif key not in sets["pregame"]:
            reason = "CAPTURE_KICKOFF_EXPIRED_OR_INVALID"
        elif key not in sets["window"]:
            reason = "CAPTURE_OUTSIDE_PUBLICATION_WINDOW"
        else:
            # Replay the evaluator's paired-history restriction without scoring.
            counts = history.groupby("timestamp").side_key.nunique()
            hist = history[history.timestamp.isin(counts[counts >= pair.side_key.nunique()].index)]
            for side in pair.side_key:
                history_counts[str(side)] = len(_build_history_points(hist[hist.side_key.eq(side)], key[2]))
            # Current evaluator drops each side separately; one surviving side
            # can still make a board market. Report that integrity concern too.
            if not any(n >= 2 for n in history_counts.values()):
                reason = "INSUFFICIENT_PARSEABLE_PAIRED_HISTORY"
        eligible = not reason
        if eligible:
            reason = "PUBLICATION_EVIDENCE_UNAVAILABLE" if board is None else "PUBLISHED" if key in published else "UNEXPLAINED_PUBLICATION_GAP"
        observed_published = key in published if board is not None else None
        warnings = []
        if len(pair) == 2:
            if pair.dk_start_iso.nunique() != 1:
                warnings.append("SIDE_KICKOFF_DISAGREEMENT")
            for column in ["bets_pct", "money_pct"]:
                numeric = pd.to_numeric(pair[column], errors="coerce")
                if not numeric.between(0, 100).all():
                    warnings.append("INVALID_" + column.upper())
            values = [_parse_snapshot_value(value, key[2])["value"] for value in pair.current_line]
            if any(v is None for v in values):
                warnings.append("UNPARSEABLE_CURRENT_LINE")
            elif key[2] == "TOTAL" and values[0] != values[1]:
                warnings.append("TOTAL_SIDE_LINE_DISAGREEMENT")
            elif key[2] == "SPREAD" and abs(sum(values)) > 1e-8:
                warnings.append("SPREAD_SIDE_LINE_DISAGREEMENT")
        results.append({**record, "dk_start_iso": kickoff_text, "window_status": window,
                        "in_window_target": target, "captured": captured if snapshots is not None else None,
                        "paired_normalized": key in sets["paired"] if snapshots is not None else None,
                        "publication_eligible": eligible if snapshots is not None else None,
                        "observed_published": observed_published,
                        "publication_conflict": bool(observed_published and not eligible),
                        "status": reason, "detail": detail,
                        "last_capture": str(history.timestamp.max()) if len(history) else "",
                        "last_paired_capture": str(pair.timestamp.max()) if len(pair) else "",
                        "history_points_by_side": json.dumps(history_counts, sort_keys=True),
                        "pair_integrity_warnings": json.dumps(warnings),
                        "partial_evaluator_side_loss": bool(history_counts and any(n < 2 for n in history_counts.values()) and any(n >= 2 for n in history_counts.values()))})
    return pd.DataFrame(results)


def summarize(rows, board_available, source_complete=False):
    targets = rows[rows.in_window_target.eq(True)]
    ready = targets[targets.publication_eligible.eq(True)]
    unknown_capture = targets.status.eq("CAPTURE_EVIDENCE_UNAVAILABLE").any()
    unresolved = rows.status.isin(["UNEXPLAINED_PUBLICATION_GAP", "CAPTURE_EVIDENCE_UNAVAILABLE",
                                  "PUBLICATION_EVIDENCE_UNAVAILABLE", "NOT_CAPTURED_OR_MARKET_NOT_NORMALIZED",
                                  "SPORT_LEAGUE_NOT_VERIFIED", "UNRESOLVED_EVENT_IDENTITY", "MISSING_OR_INVALID_KICKOFF"])
    return dict(source_census_complete=source_complete, discovered_market_records=len(rows),
                outside_publication_window=int(rows.status.eq("OUTSIDE_PUBLICATION_WINDOW").sum()),
                eligible_in_window_scope=len(targets), publication_gate_eligible=None if unknown_capture else len(ready),
                published_of_in_window_scope=int(targets.observed_published.eq(True).sum()) if board_available else None,
                published_of_gate_eligible=int(ready.observed_published.eq(True).sum()) if board_available else None,
                exclusion_counts=rows.status.value_counts().to_dict(),
                in_window_exclusion_counts=targets.status.value_counts().to_dict(),
                unexplained_publication_gaps=int(rows.status.eq("UNEXPLAINED_PUBLICATION_GAP").sum()),
                unresolved_evidence_records=int(unresolved.sum()),
                coverage_certified=bool(source_complete and not unresolved.any()))


def assert_accounted(rows):
    """Unknown evidence and unexplained gaps must fail a coverage certification."""
    evidenced_states = {"PUBLISHED", "NO_CAPTURE_IN_TWO_HOUR_WORKING_SET", "NO_SYNCHRONIZED_SIDE_PAIR",
                       "MONEYLINE_ODDS_WIDTH_NORMALIZATION_FAILURE", "INCOMPLETE_MARKET_FIELDS_OR_SIDE_COUNT",
                       "STALE_CAPTURE", "INSUFFICIENT_PARSEABLE_PAIRED_HISTORY",
                       "CAPTURE_KICKOFF_EXPIRED_OR_INVALID", "CAPTURE_OUTSIDE_PUBLICATION_WINDOW",
                       "ESPN_GAME_UNMATCHED", "SNAPSHOT_VALIDATION_REJECTED", "RAW_MARKET_PARSE_FAILED"}
    bad = rows[rows.in_window_target.eq(True) & ~rows.status.isin(evidenced_states)]
    if len(bad):
        raise AssertionError(f"{len(bad)} in-window markets lack an evidenced terminal state")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--snapshots", type=Path)
    parser.add_argument("--board", type=Path)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False) if p else None
    snapshots, board = read(args.snapshots), read(args.board)
    if not args.inventory and snapshots is None:
        parser.error("--inventory or --snapshots is required")
    inventory = read(args.inventory) if args.inventory else captured_inventory(snapshots)
    result = reconcile(inventory, snapshots, board, args.as_of)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "reconciliation.csv", index=False)
    summary = summarize(result, board is not None)
    summary.update(as_of=args.as_of, inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in [args.inventory, args.snapshots, args.board] if p})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
