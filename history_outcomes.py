"""Resolve market fixtures independently of the live Favorite ledger."""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from build_live_recent import fetch_scoreboard, game_key
from history_store import connect, now_iso, packed, store_outcome, utc
from mlb_fixtures import NEW_YORK, match_fixture, read_context, timestamp
from ufc_bout_integrity import cache_dir, match_bout, read_json


def resolve_team_game(fixture, scores):
    start = timestamp(fixture["scheduled_at"])
    key = game_key(fixture["matchup"], fixture["sport"])
    candidates = {
        item["score_provider_event_id"]: item
        for item in (scores.get(key) or scores.get("neutral:" + key) or ())
        if item.get("score_provider_event_id")
        and (provider_start := timestamp(item.get("event_time")))
        and abs(start - provider_start) <= timedelta(minutes=45)
    }
    if len(candidates) != 1:
        return "unresolved", {"reason": "missing or ambiguous paired fixture"}
    provider_id, result = next(iter(candidates.items()))
    evidence = {"provider_event_id": provider_id, "provider_start": result["event_time"],
                "status_code": result["score_status_code"]}
    if "CANCEL" in result["score_status_code"].upper():
        return "void", evidence
    if result["score_state"] != "post" or not result["score_status_code"].startswith("STATUS_FINAL"):
        return "unresolved", evidence
    if not all(str(result.get("score_" + side, "")).isdigit() for side in ("away", "home")):
        return "unresolved", evidence
    return "verified", {**evidence, "score_away": int(result["score_away"]),
                        "score_home": int(result["score_home"])}


def resolve_mlb(fixture, fixtures):
    if fixtures is None:
        return "unavailable", {"reason": "official MLB fixture feed unavailable"}
    # The live view omits cancelled fixtures. Keep that identity check while
    # letting a confirmed cancellation become a verified research void.
    outcomes = {
        game_pk: dict(game, state="void") if (
            game.get("state") == "postponed" and "cancel" in game.get("status", "").lower()
        ) else game
        for game_pk, game in fixtures.items()
    }
    final = match_fixture(outcomes, fixture["matchup"], fixture["scheduled_at"])
    if not final:
        return "unresolved", {"reason": "no unique official fixture or first-pitch evidence"}
    evidence = {key: final.get(key) for key in (
        "official_date", "scheduled_start", "first_pitch", "game_number", "doubleheader", "status",
        "innings", "scheduled_innings",
    )}
    evidence["provider_event_id"] = str(final["game_pk"])
    if final["state"] == "void":
        return "void", evidence
    if final["state"] != "post" or final["status"] not in {"Final", "Game Over", "Completed Early"}:
        return "unresolved", evidence
    return "verified", {**evidence, "score_away": final["away"]["score"],
                        "score_home": final["home"]["score"],
                        "settlement_review": final["status"] == "Completed Early"}


def resolve_ufc(fixture, bouts):
    if bouts is None:
        return "unavailable", {"reason": "bout evidence feed unavailable"}
    final = match_bout(fixture["matchup"], fixture["scheduled_at"], bouts)
    if not final:
        return "unresolved", {"reason": "no unique fighter pair and card"}
    evidence = {key: final.get(key) for key in (
        "card_id", "athlete_ids", "pair", "names", "start", "start_source", "winner", "loser",
        "result", "result_observed", "status"
    )}
    evidence["provider_event_id"] = str(final["id"])
    return {"winner": "verified", "void": "void", "push": "push"}.get(final.get("kind"), "unresolved"), evidence


def collect(db, root, now, *, day_limit):
    pending = db.execute("""
      SELECT f.*, u.checked_at FROM fixtures f
      LEFT JOIN unresolved u USING (sport,event_id)
      WHERE f.scheduled_at IS NOT NULL AND f.scheduled_at<=?
        AND NOT EXISTS (
          SELECT 1 FROM outcomes o WHERE o.sport=f.sport AND o.event_id=f.event_id
           AND o.status IN ('verified','void','push')
        )
      ORDER BY (u.checked_at IS NOT NULL),u.checked_at,f.scheduled_at DESC
    """, ((now - timedelta(hours=3)).isoformat(),)).fetchall()
    by_day = defaultdict(list)
    local_sources = {"mlb": read_context(root / "data", now),
                     "ufc": read_json(cache_dir(root / "data") / "bouts.json").get("bouts")}
    for fixture in pending:
        start = timestamp(fixture["scheduled_at"])
        if start is None:
            continue
        day = start.astimezone(NEW_YORK).date()
        sport = fixture["sport"]
        key = (sport, day)
        if sport not in local_sources and key not in by_day:
            if sum(s not in local_sources for s, _ in by_day) >= day_limit:
                continue
        by_day[key].append(fixture)
    # Reuse the existing providers for bounded correction checks. A result
    # remains available if the provider is temporarily unavailable.
    recent = defaultdict(list)
    recheck_marks = dict(db.execute("SELECT name,value FROM jobs WHERE name LIKE 'outcome_recheck:%'"))
    correction_cutoff = utc(now - timedelta(hours=6))
    for fixture in db.execute("""
        SELECT f.* FROM fixtures f WHERE f.scheduled_at BETWEEN ? AND ?
        AND EXISTS (SELECT 1 FROM outcomes o WHERE (o.sport,o.event_id)=(f.sport,f.event_id)
                    AND o.status IN ('verified','void','push'))
    """, (utc(now - timedelta(days=3)), utc(now - timedelta(hours=3)))):
        key = fixture["sport"], timestamp(fixture["scheduled_at"]).astimezone(NEW_YORK).date()
        if recheck_marks.get(f"outcome_recheck:{key[0]}:{key[1]}", "") < correction_cutoff:
            recent[key].append(fixture)
    ordered = sorted(recent, key=lambda key: (recheck_marks.get(f"outcome_recheck:{key[0]}:{key[1]}", ""), key))
    rechecked = set()
    extras = 0
    for key in ordered:
        if key not in by_day and key[0] not in local_sources:
            if extras >= 2:
                continue
            extras += 1
        by_day[key].extend(recent[key])
        rechecked.update((r["sport"], r["event_id"]) for r in recent[key])
    counts = defaultdict(int)
    for (sport, date), fixtures in by_day.items():
        if sport in local_sources:
            provider = "official_mlb_fixture_cache" if sport == "mlb" else "espn_exact_ufc_bout"
            resolver = resolve_mlb if sport == "mlb" else resolve_ufc
            scores = local_sources[sport]
            # Current MLB cache is limited to a recent window. Earlier games
            # require the imported gamePk reconciliation.
        else:
            provider = "espn_exact_team_fixture"
            midnight = datetime.combine(date, datetime.min.time(), tzinfo=timezone.utc)
            scores, availability = fetch_scoreboard(sport, now, dates=(midnight,), timeout=7)
            if availability != "available":
                scores = None
            resolver = resolve_team_game
        with db:
            for fixture in fixtures:
                event = fixture["event_id"]
                correcting = (sport, event) in rechecked
                status, evidence = resolver(fixture, scores) if scores is not None else (
                    "unavailable", {"reason": "provider unavailable"}
                )
                if correcting:
                    if status not in {"verified", "void", "push"}:
                        counts["recheck_deferred"] += 1
                        continue
                    known = db.execute("""
                        SELECT provider_event_id FROM outcomes WHERE (sport,event_id,source)=(?,?,?)
                        ORDER BY observed_at DESC LIMIT 1
                    """, (sport, event, provider)).fetchone()
                    if known and known[0] != evidence.get("provider_event_id"):
                        counts["recheck_identity_conflict"] += 1
                        continue
                    counts["rechecked"] += 1
                else:
                    counts[status] += 1
                evidence["matchup"] = fixture["matchup"]
                evidence["fixture_scheduled_at"] = fixture["scheduled_at"]
                evidence["observed_at"] = now_iso()
                if status in {"verified", "void", "push"}:
                    store_outcome(db, sport, event, provider, status, evidence, correction=True)
                    db.execute("DELETE FROM unresolved WHERE sport=? AND event_id=?", (sport, event))
                else:
                    db.execute("INSERT OR REPLACE INTO unresolved VALUES (?,?,?,?,?)",
                               (sport, event, status, evidence["observed_at"], packed(evidence)))
            if (sport, date) in recent and scores is not None:
                db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)",
                           (f"outcome_recheck:{sport}:{date}", utc(now)))
    with db:
        db.execute("DELETE FROM jobs WHERE name LIKE 'outcome_recheck:%' AND value<?",
                   (utc(now - timedelta(days=7)),))
    return dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/opt/red-fox-market-dynamics"))
    parser.add_argument("--state", type=Path, default=Path("/var/lib/redfox-history/private"))
    parser.add_argument("--day-limit", type=int, default=8)
    args = parser.parse_args()
    with connect(args.state) as db:
        print(packed({"outcome_reconciliation": collect(db, args.root, datetime.now(timezone.utc),
                                                        day_limit=args.day_limit)}))


if __name__ == "__main__":
    main()
