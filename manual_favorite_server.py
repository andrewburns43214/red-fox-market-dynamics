"""Loopback API for the single Red Fox account authorized to save overrides."""

from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

from access_verifier import access_token, bearer_token, verified_user
from build_live_recent import fetch_scoreboard, exact_scores
from manual_favorites import (csv_rows, decisions, event_rows, latest_quotes, now_utc,
                              official_board_choice, owner_board_patches, save, same_fixture, scope, snapshot_revision, stamp)
from mlb_fixtures import match_fixture, read_context
import ufc_bout_integrity as ufc
import pandas as pd
import requests

OWNER = "andrewburns43214@gmail.com"
ORIGINS = {"https://redfoxmi.com", "https://www.redfoxmi.com"}
DATA = Path(os.environ.get("REDFOX_DATA_DIR", "data"))
ENABLE = Path("/etc/redfox-manual-favorites.enabled")
ROUTE = "/api/owner/favorites"
SPORTS = {"nfl", "nba", "ncaaf", "ncaab", "nhl", "mlb", "ufc"}


def event_identity(fields):
    sport = fields.get("sport")
    event = fields.get("event")
    return (sport, event) if isinstance(sport, str) and sport in SPORTS and isinstance(event, str) and re.fullmatch(r"[0-9]{1,24}", event) else None


def check_pregame(data_dir, row):
    """Pin the verified provider identity and validate the phase on the server."""
    now = now_utc()
    sport, start = row["sport"], stamp(row["kickoff_iso"])
    if not start or (sport != "ufc" and now >= start):
        return None
    if sport == "ufc":
        enabled, _ = ufc.settings()
        if not enabled:
            return None
        bouts, notices = ufc.load_context(data_dir)
        bout = ufc.match_bout(row["game"], row["kickoff_iso"], bouts)
        candidate = next((candidate for candidate in csv_rows(Path(data_dir) / "live_recent_market_candidates.csv")
                          if scope(candidate) == scope(row) and candidate.get("market_display") == "MONEYLINE"
                          and candidate.get("game") == row["game"] and stamp(candidate.get("kickoff_iso")) == start), {})
        proof = ufc.read_market_proof(candidate)
        if not bout or proof.get("bout_id") != bout["id"] or proof.get("athlete_ids") != bout["athlete_ids"]:
            return None
        from collect_ufc_bouts import ESPN, build_bouts, get

        card_date = stamp(bout["card_at"]).astimezone(ZoneInfo("America/New_York"))
        source = get(ESPN, params={"dates": card_date.strftime("%Y%m%d"), "limit": 100}).json()
        now = now_utc()
        fresh = build_bouts(source["events"], {}, pd.Timestamp(now)).get(bout["id"])
        if (not fresh or fresh["phase"] != "pre" or fresh["athlete_ids"] != bout["athlete_ids"]
                or fresh["card_id"] != bout["card_id"] or fresh["pair"] != bout["pair"]):
            return None
        observed, actual = stamp(bout.get("observed")), stamp(bout.get("start"))
        if not observed or now - observed > ufc.MAX_PHASE_AGE or observed > now or actual or bout["phase"] != "pre":
            return None
        if ufc.risks_for(bout, notices, pd.Timestamp(now)):
            return None
        return {"id": bout["id"], "source": "ufc", "observed": now.isoformat(), "start": "",
                "pre_observed": now.isoformat(), "proof": proof}
    if sport == "mlb":
        fixture = match_fixture(read_context(data_dir, now), row["game"], start)
        if not fixture or fixture["state"] != "pre" or fixture.get("first_pitch"):
            return None
        from collect_mlb_fixtures import league_fixture, schedule_window

        day = start.astimezone(ZoneInfo("America/New_York")).date()
        current = [league_fixture(game) for day_row in schedule_window(day, day)["dates"] for game in day_row["games"]
                   if str(game.get("gamePk")) == str(fixture["game_pk"])]
        if len(current) != 1 or not current[0] or current[0]["state"] != "pre" or stamp(current[0]["scheduled_start"]) != start:
            return None
        return {"id": str(fixture["game_pk"]), "source": "mlb", "observed": now_utc().isoformat(),
                "start": start.isoformat()}
    day = start.astimezone(ZoneInfo("America/New_York"))
    scores, availability = fetch_scoreboard(sport, now, dates=(day,), timeout=4)
    if availability != "available":
        return None
    matched = {}
    for candidate in exact_scores(scores, row):
        kickoff = stamp(candidate.get("event_time"))
        identity = candidate.get("score_provider_event_id")
        if identity and kickoff and abs(kickoff - start) <= timedelta(minutes=45):
            matched[identity] = candidate
    if len(matched) != 1:
        return None
    provider_id, item = next(iter(matched.items()))
    confirmed = min(start, stamp(item["event_time"]))
    now = now_utc()
    if item["score_state"] != "pre" or item["score_status_code"] != "STATUS_SCHEDULED" or now >= confirmed:
        return None
    return {"id": provider_id, "source": "espn", "observed": now.isoformat(),
            "start": confirmed.isoformat(), "provider_start": item["event_time"]}


def event_context(data_dir, event):
    rows = event_rows(data_dir, event)
    current = decisions(data_dir, event).get(event)
    choice = official_board_choice(rows)
    if current and rows and same_fixture(rows[0], current):
        if current["action"] == "remove":
            choice = None
        elif current["action"] in {"add", "change"}:
            q = current["quote"]
            choice = {"market": q["market"], "side": q["label"], "origin": "Manual"}
    return rows, current, choice


def context_payload(data_dir, event):
    rows, current, choice = event_context(data_dir, event)
    quotes = latest_quotes(data_dir, rows, now_utc())
    return {"game": rows[0]["game"] if rows else "", "quotes": [q for q, _ in quotes.values()],
            "revision": (current or {}).get("revision", 0), "choice": choice,
            "override": (current or {}).get("action", "restore") != "restore"}


def persist(data_dir, event, owner, request):
    action, revision = request.get("action"), request.get("revision")
    reason = request.get("reason", "")
    requested_quote = request.get("quote", "")
    if (action not in ("add", "change", "remove", "restore") or type(revision) is not int
            or not 0 <= revision < 2**63 or not isinstance(reason, str) or len(reason) > 500
            or not isinstance(requested_quote, str)):
        return 400, {"error": "Invalid action."}
    rows, current, choice = event_context(data_dir, event)
    if not rows:
        return 409, {"error": "The event is unavailable or has already started."}
    if (current or {}).get("revision", 0) != revision:
        return 409, {"error": "This event has a newer decision. Review it and confirm.", "refresh": True}
    if (action == "add" and choice or action in {"change", "remove"} and not choice
            or action == "restore" and (not current or current["action"] == "restore")):
        return 409, {"error": "This action is no longer available. Review the current decision.", "refresh": True}
    if any(r.get("sport") == event[0] and r.get("event_id") == event[1]
           for r in settled_records(data_dir)):
        return 409, {"error": "The event already has a settlement record."}
    q = None
    row = rows[0]
    capture = None
    if action in {"add", "change"}:
        capture = snapshot_revision(data_dir)
        offers = latest_quotes(data_dir, rows, now_utc())
        if requested_quote not in offers:
            return 409, {"error": "The quote changed or is unavailable. Review the current prices and confirm.", "refresh": True}
        q, row = offers[requested_quote]
    fixture = check_pregame(data_dir, row)
    if not fixture or (current and same_fixture(row, current) and current["fixture"]["id"] != fixture["id"]):
        return 409, {"error": "Pregame status or fixture identity could not be verified."}
    if q:
        if fixture["source"] == "ufc" and stamp(q["observed"]) > stamp(fixture["pre_observed"]):
            return 409, {"error": "Waiting for a verified pre-bout status for this quote.", "refresh": True}
    payload = {"game": row["game"], "board_start": row["kickoff_iso"],
               "fixture": fixture, "quote": q, "row": row if q else {}}
    saved = save(data_dir, event, revision, owner, action, reason.strip(),
                 {"choice": choice, "override": (current or {}).get("action"), "revision": revision}, payload, capture)
    if saved is None:
        return 409, {"error": "The event, quote or decision changed before Save. Review the current state.", "refresh": True}
    selection = {"market": q["market"], "side": q["label"], "price": q["price"], "origin": "Manual"} if q else None
    return 200, {"revision": saved, "choice": selection, "action": action,
                 "as_of": now_utc().isoformat()}


def settled_records(data_dir):
    return csv_rows(Path(data_dir) / "performance_ledger.csv")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # Never write authorization material or private reason text.

    def reply(self, status, data):
        response = json.dumps(data, ensure_ascii=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def owner(self):
        if not ENABLE.is_file():
            self.reply(404, {"error": "Unavailable."})
            return None
        token = bearer_token(self.headers.get("Authorization")) or access_token(self.headers.get("Cookie"))
        if not token:
            self.reply(401, {"error": "Sign in required."})
            return None
        verified = verified_user(token)
        if (str(verified.get("email", "")).lower() != OWNER or not verified.get("email_confirmed_at")
                or not isinstance(verified.get("id"), str) or not verified["id"]):
            self.reply(403, {"error": "Access denied."})
            return None
        return {"id": verified["id"], "email": OWNER}

    def do_GET(self):
        path = urlsplit(self.path)
        if path.path != ROUTE:
            return self.reply(404, {"error": "Unavailable."})
        if not self.owner():
            return
        if not path.query:
            result = {"enabled": True}
            try:
                patches = owner_board_patches(DATA, now_utc())
                if patches:
                    result["favorite_patches"] = patches
            except (OSError, sqlite3.Error, ValueError, KeyError, TypeError):
                # A transient read failure must not disable legitimate owner access.
                pass
            return self.reply(200, result)
        query = parse_qs(path.query)
        event = event_identity({key: values[0] for key, values in query.items() if len(values) == 1})
        if not event:
            return self.reply(400, {"error": "Invalid event."})
        try:
            self.reply(200, context_payload(DATA, event))
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, requests.RequestException):
            self.reply(503, {"error": "Favorite controls are temporarily unavailable."})

    def do_POST(self):
        if self.path != ROUTE:
            return self.reply(404, {"error": "Unavailable."})
        if (self.headers.get("Origin") not in ORIGINS or
                self.headers.get("Sec-Fetch-Site", "same-origin") != "same-origin"):
            return self.reply(403, {"error": "Invalid request origin."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if self.headers.get_content_type() != "application/json" or not 0 < length <= 4096:
            return self.reply(400, {"error": "Invalid request."})
        owner = self.owner()
        if not owner:
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeError):
            payload = None
        event = event_identity(payload) if isinstance(payload, dict) else None
        if not event:
            return self.reply(400, {"error": "Invalid event."})
        try:
            self.reply(*persist(DATA, event, owner, payload))
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, requests.RequestException):
            self.reply(503, {"error": "Favorite controls are temporarily unavailable."})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("MANUAL_FAVORITE_PORT", "5054"))), Handler).serve_forever()
