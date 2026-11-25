"""Read independently reversible NHL-only controls without restarting jobs."""
import json
import os
from pathlib import Path

OVERRIDE = Path("/etc/redfox-nhl-release.json")


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
