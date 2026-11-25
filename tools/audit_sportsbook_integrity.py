"""Reconcile copied sportsbook exports without writing to the source directory."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canonical_match import build_canonical_key_from_dk
from red_fox_favorite import _side_identity


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def timestamp(value):
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return observed.astimezone(timezone.utc) if observed.tzinfo else None
    except (AttributeError, ValueError):
        return None


def market_key(row):
    return (row["sport"], row.get("game_id") or row.get("event_id"),
            row.get("market_display") or row.get("market"))


def state_of(row):
    return str(row.get("red_fox_favorite") or row.get("favorite_qualified") or "").lower() in {"yes", "true", "1"}


def build_reference_event_map(board, references):
    """Map exact teams and UTC scheduled instant, explicitly rejecting collisions."""
    identities = defaultdict(set)
    for row in board:
        kickoff = timestamp(row.get("kickoff_iso") or row.get("dk_start_iso"))
        if kickoff is None:
            continue
        canonical = build_canonical_key_from_dk(row["game"], row["sport"], kickoff.isoformat())
        if canonical:
            identities[(canonical, kickoff)].add((row["sport"], row["game_id"]))
    matches, ambiguous = {}, set()
    for row in references:
        key = row["canonical_key"], timestamp(row.get("commence_time"))
        candidates = identities.get(key, ())
        if len(candidates) == 1:
            matches[key] = next(iter(candidates))
        elif len(candidates) > 1:
            ambiguous.add(key)
    return matches, ambiguous


def inspect_review_history(rows):
    by_key, incidents = defaultdict(list), []
    counts = Counter()
    for row in sorted(rows, key=lambda item: item["recorded_at"]):
        key = market_key(row)
        preceding = by_key[key]
        if row["review_state"] == "applied" and row["review_action"] == "addition":
            counts["applied_additions"] += 1
            if preceding and preceding[-1]["review_state"] == "pending" and preceding[-1]["review_action"] == "addition":
                now = timestamp(row["recorded_at"])
                last = preceding[-1]
                last_at = timestamp(last["recorded_at"])
                if now and last_at:
                    gap = (now - last_at).total_seconds() / 60
                    counts["adjacent_pending_checked"] += 1
                    pending_side, applied_side = (_side_identity(item["favorite_side"]) for item in (last, row))
                    if not pending_side or not applied_side:
                        counts["missing_side_on_applied_or_pending"] += 1
                    elif pending_side != applied_side:
                        counts["adjacent_pending_side_differs"] += 1
                        incidents.append({"market": key, "pending": last, "applied": row, "gap_minutes": gap})
        if row["review_state"] == "pending":
            counts["pending_rows"] += 1
            if not row.get("favorite_rule_version") or not row.get("source_observed_at"):
                counts["pending_missing_version_or_source_timestamp"] += 1
        preceding.append(row)
    return {"counts": counts, "adjacent_side_differences": incidents,
            "interpretation": "A different adjacent pending side is an exposure, not proof of causality."}


def capture_state_conflicts(board, live, tracking, ledger):
    tracking_by_key = defaultdict(list)
    for row in sorted(tracking, key=lambda entry: entry["recorded_at"]):
        tracking_by_key[market_key(row)].append(row)
    frozen_ledger = {market_key(row): row for row in ledger}
    result = {}
    for name, display in (("board", board), ("live_recent", live)):
        mismatches = []
        counts = Counter(display_rows=len(display))
        for row in display:
            key = market_key(row)
            shown = state_of(row)
            history = tracking_by_key.get(key, [])
            if name == "live_recent":
                start = timestamp(row.get("kickoff_iso"))
                if start:
                    history = [entry for entry in history if
                               (captured := timestamp(entry["recorded_at"])) and captured <= start]
            best = history[-1] if history else None
            if best:
                counts["matched_tracking"] += 1
                tracked = best["favorite_state"] == "qualified"
                if shown != tracked:
                    counts["display_differs_from_latest_tracking"] += 1
                    mismatches.append({"key": key, "visible": shown, "tracking": tracked,
                                       "display_reason": row.get("favorite_reason"),
                                       "tracking_at": best["recorded_at"]})
            settled = frozen_ledger.get(key)
            if settled:
                counts["matched_ledger"] += 1
                if shown != state_of(settled):
                    counts["display_differs_from_ledger"] += 1
        result[name] = {"counts": counts, "tracking_differences": mismatches}
    return result


def dump(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--stage-one", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original_ledger = args.original / "performance_ledger.csv"
    ledger = records(original_ledger)
    favorite = [row for row in ledger if state_of(row)]
    verified = json.loads((args.stage_one / "graded_reads.json").read_text(encoding="utf-8"))["favorites"]
    review = records(args.snapshot / "red_fox_favorite_review.csv")
    board = records(args.snapshot / "anomaly_board.csv")
    live_recent = records(args.snapshot / "live_recent.csv")
    tracking = records(args.snapshot / "red_fox_favorite_tracking.csv")
    recent_ledger = records(args.snapshot / "performance_ledger.csv")
    dump(args.output / "original.json", {
        "source": str(original_ledger), "sha256": hashlib.sha256(original_ledger.read_bytes()).hexdigest(),
        "official_grade": Counter(row["grade"] or "pending" for row in favorite),
        "official_by_sport": {sport: Counter(row["grade"] or "pending" for row in favorite if row["sport"] == sport)
                              for sport in sorted({row["sport"] for row in favorite})},
        "entry_sources": Counter(row.get("decision_line_source") or "unknown" for row in favorite),
    })
    dump(args.output / "independently_verified.json", verified)
    dump(args.output / "unresolved.json", {
        "review": inspect_review_history(review),
        "publication_reconciliation": capture_state_conflicts(board, live_recent, tracking, recent_ledger),
        "notes": [
            "Snapshot tables were copied during normal operation and can differ transiently across workers.",
            "Legacy pending reviews lack rule versions and source timestamps. A reliable counterfactual record requires them.",
            "A verified opposite-side pending review exposure is not a retroactive grade or an official correction.",
            "Archived close values without explicit time or source cannot be promoted to verified closes.",
        ],
    })


if __name__ == "__main__":
    main()
