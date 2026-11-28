"""Attach evidence already published by Team Intelligence to a private pregame receipt."""

from datetime import datetime, timezone
import json
from pathlib import Path

from history_store import utc


class TeamEvidence:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.cache = {}

    def capture(self, sport, event, observed, scheduled):
        if sport not in {"nfl", "ncaaf", "nba", "ncaab", "nhl", "mlb", "ufc"}:
            return None
        path = self.directory / (sport + ".json")
        try:
            stat = path.stat()
            available = utc(datetime.fromtimestamp(stat.st_mtime, timezone.utc))
            if available > observed or stat.st_size > 16 * 1024**2:
                return None
            if self.cache.get(sport, (None,))[0] != stat.st_mtime_ns:
                document = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(document, dict) or path.stat().st_mtime_ns != stat.st_mtime_ns:
                    return None
                if len(self.cache) >= 2 and sport not in self.cache:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[sport] = stat.st_mtime_ns, document
            else:
                document = self.cache[sport][1]
        except (OSError, ValueError, TypeError):
            return None
        generated = utc(document.get("generated"))
        fixture = (document.get("fixtures") or {}).get(event) or {}
        if not generated or generated > observed or utc(fixture.get("start")) != scheduled:
            return None
        games = document.get("games") or {}
        recent = {}
        for identity in fixture.get("keys", ()):
            records = []
            for uid in (document.get("teams") or {}).get(identity, ()):
                game = games.get(uid) or {}
                verified = utc(game.get("source_at"))
                started = utc(game.get("start"))
                if started and started < observed and verified and verified <= observed and game.get("scores"):
                    records.append({key:game.get(key) for key in ("names", "keys", "start", "scores", "source", "source_at")})
                if len(records) == 5:
                    break
            if records:
                recent[identity] = records
        sporting = fixture.get("sporting") or {}
        # Store only prepublished fixture evidence and timestamped final scores.
        # The copy belongs to private research and never changes market forecasts.
        if not sporting and not recent:
            return None
        return {"available_at": available, "generated": generated, "fixture": event,
                "sporting": sporting if len(json.dumps(sporting)) <= 64 * 1024 else {}, "verified_recent": recent}
