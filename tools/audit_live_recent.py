"""Run a production-shaped score refresh and prove frozen fields are unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_live_recent as live


SCORE_FIELDS = {
    "score_away", "score_home", "score_status", "score_state", "score_provider",
    "score_provider_event_id", "score_match_state", "score_updated_at_utc", "score_completed_at_utc",
    "event_time",
}


def stable_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    normalized = frame.loc[:, columns].fillna("").astype(str).sort_values(columns, kind="mergesort")
    return hashlib.sha256(normalized.to_csv(index=False).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before_path = args.output_dir / "live_recent_before.csv"
    after_path = args.output_dir / "live_recent_after.csv"
    coverage_path = args.output_dir / "live_score_coverage.json"
    shutil.copyfile(args.source, before_path)
    shutil.copyfile(args.source, after_path)

    live.OUT = after_path
    live.SCORE_COVERAGE_OUT = coverage_path
    live.main(scores_only=True)

    before = pd.read_csv(before_path, dtype=str, keep_default_na=False)
    after = pd.read_csv(after_path, dtype=str, keep_default_na=False)
    keys = [column for column in ("sport", "game_id", "market_display", "flagged_side") if column in before and column in after]
    frozen = sorted((set(before.columns) & set(after.columns)) - SCORE_FIELDS)
    common = before.merge(after[keys].drop_duplicates(), on=keys, how="inner")
    after_common = after.merge(before[keys].drop_duplicates(), on=keys, how="inner")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    report = {
        "before_rows": len(before),
        "after_rows": len(after),
        "before_games": int(before[["sport", "game_id", "game"]].drop_duplicates().shape[0]),
        "after_games": int(after[["sport", "game_id", "game"]].drop_duplicates().shape[0]),
        "frozen_columns_compared": frozen,
        "frozen_hash_before": stable_hash(common, frozen),
        "frozen_hash_after": stable_hash(after_common, frozen),
        "coverage": coverage,
    }
    report["frozen_unchanged"] = report["frozen_hash_before"] == report["frozen_hash_after"]
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["frozen_unchanged"]:
        raise SystemExit("Frozen Live & Recent fields changed during score refresh")


if __name__ == "__main__":
    main()
