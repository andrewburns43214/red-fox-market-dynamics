"""Read independently reversible NHL-only controls without restarting jobs."""
import json
import os
from pathlib import Path

OVERRIDE = Path("/etc/redfox-nhl-release.json")
OFFICIAL_START = Path("/etc/redfox-nhl-official-since")
OFFICIAL_FLAGS = {
    "public_backed_moneyline": "RED_FOX_NHL_OFFICIAL_PUBLIC",
    "genuine_crossover": "RED_FOX_NHL_OFFICIAL_CROSSOVER",
}
OFFICIAL_VERSIONS = {pathway: f"red_fox_favorite_nhl_{pathway}_v1" for pathway in OFFICIAL_FLAGS}
OFFICIAL_HANDOFF = "nhl_replacement_verified_handoff"


def switches():
    try:
        values = json.loads(OVERRIDE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        print(f"NHL release settings unavailable: {type(exc).__name__}", flush=True)
        return {"RED_FOX_NHL_PAUSE_ORDINARY": True, "RED_FOX_NHL_PAUSE_HIGH_HANDLE": True}
    if not isinstance(values, dict) or any(type(v) is not bool for v in values.values()):
        print("NHL release settings must be a JSON object of booleans.", flush=True)
        return {"RED_FOX_NHL_PAUSE_ORDINARY": True, "RED_FOX_NHL_PAUSE_HIGH_HANDLE": True}
    return values


def enabled(flag, settings=None):
    value = os.getenv(flag)
    if value is not None:
        return value.lower() in ("1", "true", "yes", "on")
    return (switches() if settings is None else settings).get(flag, False)


def official_start():
    try:
        return OFFICIAL_START.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def official_nhl_favorite(row):
    if row is None or row.get("sport") != "nhl" or row.get("market_display") != "MONEYLINE":
        return False
    version = OFFICIAL_VERSIONS.get(row.get("favorite_pathway"))
    return bool(version and row.get("favorite_rule_version") == version)
