"""Fail closed when the customer-facing prop projection feed is stale or degraded."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from prop_projection import parse_time
from prop_projection_config import SPORTS


def _source_age_limit(sport: str, lead_minutes: float) -> float:
    """Maximum acceptable oldest input age, including normal cron jitter."""
    if sport in {"nfl", "ncaaf"}:
        if lead_minutes <= 60:
            return 25
        if lead_minutes <= 360:
            return 45
        if lead_minutes <= 1440:
            return 75
        if lead_minutes <= 2880:
            return 135
        return 195
    if lead_minutes <= 60:
        return 30
    if lead_minutes <= 360:
        return 45
    return 75


def inspect_prop_health(path: Path, now: datetime | None = None, max_payload_age_minutes: float = 20) -> dict:
    now = now or datetime.now(timezone.utc)
    issues: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"ok": False, "issues": ["prop payload missing"], "projection_count": 0}
    except (OSError, ValueError, TypeError) as error:
        return {"ok": False, "issues": [f"prop payload unreadable:{type(error).__name__}"], "projection_count": 0}

    generated = parse_time(payload.get("generated_at"))
    if not generated:
        issues.append("prop payload generated_at missing")
        payload_age = None
    else:
        payload_age = max(0.0, (now - generated).total_seconds() / 60)
        if payload_age > max_payload_age_minutes:
            issues.append(f"prop payload stale:{payload_age:.1f}m")

    if payload.get("collection_status") != "OK":
        failures = payload.get("failed_sports") or []
        detail = ",".join(str(item.get("sport") or "unknown") for item in failures if isinstance(item, dict))
        issues.append("prop upstream degraded" + (f":{detail}" if detail else ""))

    projections = payload.get("projections")
    if not isinstance(projections, list):
        issues.append("prop projections is not a list")
        projections = []

    checked = 0
    for item in projections:
        if not isinstance(item, dict):
            issues.append("malformed prop projection")
            continue
        sport = str(item.get("sport") or "").lower()
        config = SPORTS.get(sport, {})
        expected = config.get("public_model_version")
        if expected and item.get("model_version") != expected:
            issues.append(f"{sport or 'unknown'} wrong model:{item.get('model_version') or 'missing'}")
        start = parse_time(item.get("commence_time"))
        if not start or start <= now or item.get("status") != "AVAILABLE":
            continue
        checked += 1
        event = str(item.get("event_id") or "unknown")
        for field in ("away_score", "home_score"):
            try:
                value = float(item.get(field))
                if not math.isfinite(value) or value < 0:
                    raise ValueError
            except (TypeError, ValueError):
                issues.append(f"{sport}:{event} invalid {field}")
        try:
            source_age = float(item.get("oldest_observation_age_minutes"))
        except (TypeError, ValueError):
            issues.append(f"{sport}:{event} source age missing")
            continue
        lead = max(0.0, (start - now).total_seconds() / 60)
        limit = _source_age_limit(sport, lead)
        if source_age > limit:
            issues.append(f"{sport}:{event} source stale:{source_age:.1f}m>{limit:.0f}m")

    return {
        "ok": not issues,
        "issues": issues,
        "generated_at": payload.get("generated_at"),
        "payload_age_minutes": None if payload_age is None else round(payload_age, 1),
        "collection_status": payload.get("collection_status"),
        "projection_count": len(projections),
        "upcoming_available_checked": checked,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("data/prop_projections.json"))
    parser.add_argument("--max-payload-age-minutes", type=float, default=20)
    args = parser.parse_args(argv)
    report = inspect_prop_health(args.path, max_payload_age_minutes=args.max_payload_age_minutes)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
