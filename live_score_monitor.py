"""Read-only health check for the Live & Recent score coverage artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def evaluate(path: Path, max_age_minutes: int = 5) -> tuple[dict, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    generated = datetime.fromisoformat(str(payload["generated_at_utc"]).replace("Z", "+00:00"))
    age_minutes = (datetime.now(timezone.utc) - generated.astimezone(timezone.utc)).total_seconds() / 60
    issues = []
    if age_minutes > max_age_minutes:
        issues.append(f"coverage artifact stale ({age_minutes:.1f}m)")
    for field in ("unmatched", "stale", "provider_unavailable"):
        count = int(payload.get(field, 0))
        if count:
            issues.append(f"{field.replace('_', ' ')} {count}")
    return payload, issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-age-minutes", type=int, default=5)
    args = parser.parse_args()
    path = args.data_dir / "live_score_coverage.json"
    try:
        payload, issues = evaluate(path, args.max_age_minutes)
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "alert", "issues": [f"coverage unavailable: {type(error).__name__}"]}))
        raise SystemExit(1)
    output = {
        "status": "alert" if issues else "ok",
        "issues": issues,
        "active_live_games": payload.get("active_live_games", 0),
        "matched": payload.get("matched", 0),
        "receiving_score": payload.get("receiving_score", 0),
        "unmatched": payload.get("unmatched", 0),
        "stale": payload.get("stale", 0),
        "provider_unavailable": payload.get("provider_unavailable", 0),
    }
    print(json.dumps(output, sort_keys=True))
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
