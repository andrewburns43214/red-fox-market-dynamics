"""Materialize auditable historical decision windows for private research."""

import argparse
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import time

from history_blocks import decode, encode
from history_store import FREE_DISK_RESERVE, connect, eligible_pairs, latest_provider_outcomes, number, packed, start_evidence, utc
from team_aliases import _split_game, normalize_team_name
from ufc_bout_integrity import fighter_key


ADDED_WINDOWS = (2160, 960, 600, 240, 180, 90)
WINDOWS = tuple(sorted((4320, 2880, 1440, 360, 120, 60, 40, 30, 20, 10, *ADDED_WINDOWS), reverse=True))
MAX_AGE = timedelta(minutes=20)
MARKETS = ("SPREAD", "MONEYLINE", "TOTAL")
CONTEXT_VERSION = 5


def side_key(value, sport, market):
    value = str(value or "").strip()
    if market == "TOTAL":
        return {"under": "under", "u": "under", "over": "over", "o": "over"}.get(
            value.split(" ", 1)[0].casefold(), "")
    if market == "SPREAD":
        value = re.sub(r"\s[+-]?\d+(?:\.\d+)?(?:\s*\([^)]*\))?\s*$", "", value)
    return fighter_key(value) if sport == "ufc" else normalize_team_name(value, sport)


def fixture_pairs(pairs, sport, market, names):
    expected = {side_key(name, sport, market) for name in (("Over", "Under") if market == "TOTAL" else names)}
    if len(expected) != 2 or "" in expected:
        return []
    return [(clock, tuple(dict(row, side=identity) for row, identity in zip(pair, keys)))
            for clock, pair in pairs
            if set(keys := [side_key(row["side"], sport, market) for row in pair]) == expected]


def implied(price):
    if price is None or abs(price) < 100:
        return None
    return abs(price) / (100 + abs(price)) if price < 0 else 100 / (100 + price)


def delta(current, opening):
    return current - opening if current is not None and opening is not None else None


def cross_evidence(market, quote, previous=()):
    line = delta(quote["open_line"], quote["line"])
    price = delta(implied(quote["price"]), implied(quote["open_price"]))
    threshold = .005 if market == "MONEYLINE" else .25
    value = price if market == "MONEYLINE" else line
    before = [delta(implied(prior["price"]), implied(prior["open_price"])) if market == "MONEYLINE"
              else delta(prior["open_line"], prior["line"]) for prior in previous]
    sustained = value is not None and value >= threshold and None not in before and all(
        point >= threshold for point in before
    ) if before else None
    return {"market": market, "line_toward": line, "price_toward_implied": price,
            "sustained_toward": sustained}


def recorded_split_context(db, sport, event, market, before):
    """Use recorded transitions only. Older unaudited cases stay unknown."""
    if market == "TOTAL":
        return {}
    lower = utc(datetime.fromisoformat(before) - timedelta(minutes=45))
    rows = db.execute("""
        SELECT evidence FROM decisions
        WHERE (sport,event_id,market)=(?,?,?) AND known_at BETWEEN ? AND ?
          AND source IN ('red_fox_favorite_review','red_fox_favorite_tracking')
        ORDER BY known_at DESC,digest
    """, (sport, event, market, lower, before))
    seen = {}
    for row in rows:
        record = json.loads(row["evidence"])
        try:
            proof = json.loads(record.get("decision_evidence") or "{}")
        except ValueError:
            continue
        if not isinstance(proof, dict):
            continue
        failed = record.get("review_state") == "cleared" or record.get("favorite_state") == "not_qualified"
        category, phase = proof.get("split_breach"), proof.get("phase")
        quoted = utc(proof.get("source_observed_at"))
        if not failed or category not in {"ticket_only", "money_only", "both"} or not quoted or quoted > before:
            continue
        name = proof.get("selected_side") or record.get("favorite_side") or ""
        if market == "SPREAD":
            name = name.rsplit(" ", 1)[0] if len(name.split()) > 1 and number(name.rsplit(" ", 1)[1]) is not None else name
        identity = fighter_key(name) if sport == "ufc" else normalize_team_name(name, sport)
        seen.setdefault(identity, {"split_breach": category, "phase": phase})
    return seen


def features(history, entry, market):
    opening, current = implied(entry["open_price"]), implied(entry["price"])
    line_move = delta(entry["line"], entry["open_line"])
    direction = 1 if market == "TOTAL" and entry["side"] == "over" else -1
    if market == "MONEYLINE":
        path = [implied(row["price"]) for row in history]
        anchor = opening
    else:
        path = [direction * row["line"] if row["line"] is not None else None for row in history]
        anchor = direction * entry["open_line"] if entry["open_line"] is not None else None
    valid_path = [(row, value) for row, value in zip(history, path) if value is not None]
    turn = next((row for row, value in valid_path if anchor is not None and
                 value - anchor >= (.005 if market == "MONEYLINE" else .25)), {})
    path = [value for _, value in valid_path]
    steps = [1 if b > a else -1 for a, b in zip(path, path[1:]) if a != b]
    first = history[0] if history else {}
    def opening_split(key):
        value = first.get(key)
        return value if len(history) > 1 and value is not None and 0 < value < 100 else None

    initial_tickets, initial_money = opening_split("tickets"), opening_split("money")
    start, end = utc(first.get("captured_at")), utc(entry.get("captured_at"))
    threshold = .005 if market == "MONEYLINE" else .25
    retreat = max(path) - path[-1] if path else None
    rebound = path[-1] - min(path) if path else None
    move = path[-1] - anchor if path and anchor is not None else None
    # Measure whether price held during the interval in which the split grew.
    # Use the opening observation as the comparison baseline, never an outcome.
    tone = delta(current, opening) if market == "MONEYLINE" else (
        direction * line_move if line_move is not None else None)
    price_move = delta(current, opening)
    support_held = (tone >= -threshold and retreat < threshold and (abs(tone) >= threshold or price_move >= -.005)
                    if tone is not None and price_move is not None and retreat is not None else None)
    stance = ("toward" if move >= threshold else "against" if move <= -threshold else "steady") if move is not None else None
    trajectory = None
    if len(path) >= 2 and stance:
        trajectory = ("retraced" if retreat >= threshold and max(path) - anchor >= threshold else
                      "recovered" if rebound >= threshold and (min(path) < anchor - threshold or -1 in steps) else
                      stance)
    turns = []
    if trajectory in {"retraced", "recovered"} and valid_path[:-1]:
        for extreme, value in (min(valid_path[:-1], key=lambda item: item[1]), max(valid_path[:-1], key=lambda item: item[1])):
            if abs(value - anchor) >= threshold and abs(value - path[-1]) >= threshold:
                turns.append({key: extreme[key] for key in ("line", "price", "captured_at")})
        turns.sort(key=lambda row: row["captured_at"])
    return {
        "side": entry["side"], "line": entry["line"], "price": entry["price"],
        "open_line": entry["open_line"], "open_price": entry["open_price"],
        "tickets": entry["tickets"], "money": entry["money"],
        "line_toward": None if line_move is None else line_move * direction,
        "price_toward_implied": delta(current, opening),
        "path_max_toward": max(path) - anchor if path and anchor is not None else None,
        "path_min_toward": min(path) - anchor if path and anchor is not None else None,
        "path_retrace": retreat,
        "path_direction_changes": sum(a != b for a, b in zip(steps, steps[1:])),
        "opening_tickets": initial_tickets, "opening_money": initial_money,
        "ticket_change": delta(entry["tickets"], initial_tickets),
        "money_change": delta(entry["money"], initial_money),
        "tickets_after_move": delta(entry["tickets"], turn.get("tickets")),
        "money_after_move": delta(entry["money"], turn.get("money")),
        "split_gap": delta(entry["money"], entry["tickets"]),
        "observations": len(history),
        "stance": stance, "trajectory": trajectory, "path_turns": turns,
        "ticket_growth_price_held": entry["tickets"] > initial_tickets if support_held and
        initial_tickets is not None and entry["tickets"] is not None else None,
        "money_growth_price_held": entry["money"] > initial_money if support_held and
        initial_money is not None and entry["money"] is not None else None,
        "duration_minutes": (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60
        if start and end else None,
    }


def latest(pairs, timestamps, deadline):
    at = bisect_right(timestamps, deadline) - 1
    if at >= 0 and datetime.fromisoformat(deadline) - datetime.fromisoformat(timestamps[at]) <= MAX_AGE:
        return at, pairs[at][1]
    return None, None


def sided_history(pairs, index, side):
    return [row for _, pair in pairs[:index + 1] for row in pair if row["side"] == side]


def aligned_capture(pairs, clocks, cutoff, schedule):
    """A secondary market can only contribute a paired quote already known."""
    index, pair = latest(pairs, clocks, cutoff)
    if pair is None or pair[0]["scheduled_at"] != schedule:
        return None
    return index, pair


def aligned_game(markets, clocks, primary, cutoff, schedule, values, *, vectors=None):
    context = {}
    if vectors is None:
        vectors = {}
    for market, pairs in markets.items():
        if market == primary or not (capture := aligned_capture(pairs, clocks[market], cutoff, schedule)):
            continue
        index, pair = capture
        key = market, pair[0]["captured_at"]
        if key not in vectors:
            vectors[key] = aligned_values(pairs, index, pair, market, schedule)
        context[market] = vectors[key]
    reference = None
    if primary == "TOTAL":
        # Compare equivalent team roles. The reference team is known before
        # the decision and never selected using the eventual score.
        for market, field in (("MONEYLINE", "open_price"), ("SPREAD", "open_line")):
            quotes = [q for q in context.get(market, {}).values() if
                      (implied(q[field]) if market == "MONEYLINE" else q[field]) is not None]
            if len(quotes) == 2:
                ranked = sorted(quotes, key=lambda q: -implied(q[field]) if market == "MONEYLINE" else q[field])
                if ranked[0][field] != ranked[1][field]:
                    reference = ranked[0]["side"]
                    break
    for value in values:
        value["game_context"] = {
            market: quotes[key] for market, quotes in context.items()
            if (key := "over" if market == "TOTAL" else reference if primary == "TOTAL" else value["side"]) in quotes
        }
        if reference:
            value["context_reference"] = reference
        paired = "MONEYLINE" if primary == "SPREAD" else "SPREAD" if primary == "MONEYLINE" else None
        if paired in value["game_context"]:
            other = value["game_context"][paired]
            value["cross_market"] = {k: other[k] for k in ("market", "line_toward", "price_toward_implied", "sustained_toward")}


def aligned_values(pairs, index, pair, market, schedule):
    result = {}
    for quote in pair:
        tape = sided_history(pairs, index, quote["side"])
        full = features(tape, quote, market)
        recent_start = utc(datetime.fromisoformat(quote["captured_at"]) - timedelta(minutes=30))
        recent = [row for row in tape[-3:-1] if row["captured_at"] >= recent_start and
                  row["scheduled_at"] == schedule]
        fields = ("side", "line", "price", "open_line", "open_price", "tickets", "money",
                  "line_toward", "price_toward_implied", "path_retrace", "path_direction_changes",
                  "stance", "trajectory", "path_turns")
        result[quote["side"]] = {key: full[key] for key in fields} | {
            "market": market, "captured_at": quote["captured_at"],
            "sustained_toward": cross_evidence(market, quote, recent)["sustained_toward"] if market != "TOTAL" else None,
        }
    return result


def generate(db, sport, event):
    actual = start_evidence(db, sport, event)
    fixture = db.execute("SELECT matchup FROM fixtures WHERE (sport,event_id)=(?,?)", (sport, event)).fetchone()
    names = _split_game(fixture[0]) if fixture else ("", "")
    markets = {
        market: fixture_pairs(eligible_pairs(db, sport, event, market, "9999", actual_start=actual),
                              sport, market, names)
        for market in (("MONEYLINE",) if sport == "ufc" else MARKETS)
    }
    clocks = {market: [capture for capture, _ in pairs] for market, pairs in markets.items()}
    known_match = {}
    vectors = {}
    for market, pairs in markets.items():
        schedules = {pair[0]["scheduled_at"] for _, pair in pairs}
        for minutes in WINDOWS:
            for schedule in sorted(schedules, reverse=True):
                deadline = utc(datetime.fromisoformat(schedule) - timedelta(minutes=minutes))
                if actual and deadline >= actual:
                    continue
                index, pair = latest(pairs, clocks[market], deadline)
                if pair is None or pair[0]["scheduled_at"] != schedule:
                    continue

                # Earlier quotes with a different schedule remain part of the
                # same fixture's movement history. Later quotes never enter it.
                values = [features(sided_history(pairs, index, row["side"]), row, market) for row in pair]
                # Anchor every secondary capture to the primary price timestamp.
                # A peer quote between that timestamp and the window is lookahead.
                aligned_game(markets, clocks, market, pair[0]["captured_at"], schedule, values, vectors=vectors)
                audits = recorded_split_context(db, sport, event, market, pair[0]["captured_at"])
                for value in values:
                    if value["side"] in audits:
                        value["split_audit"] = audits[value["side"]]
                peer = next((q for q in values[0]["game_context"].values() if q["market"] != "TOTAL"), {})
                cross_at = peer.get("captured_at")
                context = db.execute("""
                    SELECT known_at,digest,evidence FROM decisions
                    WHERE (sport,event_id,market)=(?,?,?) AND source='published_board' AND known_at<=?
                    ORDER BY known_at DESC,digest LIMIT 1
                """, (sport, event, market, pair[0]["captured_at"])).fetchone()
                if context:
                    proof = json.loads(context["evidence"])
                    for side in values:
                        evidence = next((x for x in proof.get("sides", [])
                                         if x.get("flagged_side", "").casefold() in {
                                             side["side"], f'{side["side"]} {side["line"]:+g}'
                                             if side["line"] is not None else side["side"],
                                         }), {})
                        side["context"] = {key: evidence[key] for key in (
                            "reaction", "path", "data_badge", "evidence_role", "whipsaw_recovered",
                            "context_chips", "response_direction",
                        ) if key in evidence}
                known_match[market, minutes] = (
                    sport, market, minutes, event, deadline, pair[0]["captured_at"], schedule,
                    encode({"sides": values, "cross_entry_at": cross_at, "verified_start": actual,
                            "published_evidence": {"known_at": context["known_at"], "digest": context["digest"]}
                            if context else None, "version": CONTEXT_VERSION}),
                )
                break
    return known_match.values()


def rebuild(db, now, limit, *, upgrade_only=False, upgrade_sport=None, budget=None):
    db.execute("""CREATE TABLE IF NOT EXISTS comparison_context_ready(
        sport TEXT,event_id TEXT,version INTEGER,PRIMARY KEY(sport,event_id)) WITHOUT ROWID""")
    events = [] if upgrade_only else db.execute("""
        SELECT f.sport,f.event_id FROM comparison_pending p JOIN fixtures f USING(sport,event_id)
        WHERE f.scheduled_at<=? OR (f.scheduled_at IS NULL AND f.last_at<=?)
        ORDER BY f.last_at DESC LIMIT ?
    """, (utc(now - timedelta(hours=3)), utc(now - timedelta(days=1)), max(1, limit * 3 // 4))).fetchall()
    upgrade = db.execute("""
        WITH eligible AS (
            SELECT w.sport,w.event_id,max(w.scheduled_at) AS scheduled FROM comparison_windows w
            LEFT JOIN comparison_context_ready v USING(sport,event_id)
            WHERE coalesce(v.version,0)<? AND (? IS NULL OR w.sport=?) AND NOT EXISTS (
                SELECT 1 FROM comparison_pending p WHERE p.sport=w.sport AND p.event_id=w.event_id)
            GROUP BY w.sport,w.event_id
        ), ranked AS (
            SELECT *, row_number() OVER (PARTITION BY sport ORDER BY scheduled DESC,event_id) AS position
            FROM eligible
        )
        SELECT sport,event_id FROM ranked ORDER BY position,scheduled DESC,sport LIMIT ?
    """, (CONTEXT_VERSION, upgrade_sport, upgrade_sport, max(0, limit - len(events)))).fetchall()
    queue = []
    for index in range(max((len(events) + 2) // 3, len(upgrade))):
        if index < len(upgrade):
            queue.append((True, upgrade[index]))
        queue.extend((False, event) for event in events[index * 3:index * 3 + 3])
    deadline = time.monotonic() + budget if budget is not None else None
    fixtures = upgraded = windows = 0
    replaced = 0
    processed = 0
    for is_upgrade, (sport, event) in queue:
        if deadline is not None and time.monotonic() >= deadline:
            break
        processed += 1
        generated = generate(db, sport, event)
        with db:
            if upgrade_only:
                db.execute("BEGIN IMMEDIATE")
                if db.execute("""
                    SELECT 1 FROM comparison_pending WHERE (sport,event_id)=(?,?)
                    UNION ALL SELECT 1 FROM comparison_context_ready
                    WHERE (sport,event_id,version)=(?,?,?) LIMIT 1
                """, (sport, event, sport, event, CONTEXT_VERSION)).fetchone():
                    continue
            replaced += db.execute("DELETE FROM comparison_windows WHERE (sport,event_id)=(?,?)",
                                   (sport, event)).rowcount
            db.executemany("INSERT INTO comparison_windows VALUES (?,?,?,?,?,?,?,?)", generated)
            db.execute("DELETE FROM comparison_pending WHERE (sport,event_id)=(?,?)", (sport, event))
            db.execute("INSERT OR REPLACE INTO comparison_context_ready VALUES (?,?,?)",
                       (sport, event, CONTEXT_VERSION))
        windows += len(generated)
        fixtures += 1
        upgraded += is_upgrade
    if windows or replaced:
        with db:
            db.execute("INSERT OR REPLACE INTO jobs VALUES ('comparison_generation',?)", (utc(now),))
    return {"fixtures": fixtures, "windows": windows, "context_upgraded": upgraded,
            "budget_deferred": len(queue) - processed}


def grade(sport, market, side, state, result):
    if state in {"push", "void"}:
        return state
    if state != "verified" or result.get("settlement_review"):
        return None
    if sport == "ufc":
        winner = fighter_key(result.get("winner") or "")
        competitors = {fighter_key(name) for name in _split_game(result.get("matchup"))}
        selected = fighter_key(side["side"])
        if winner and winner in competitors and selected in competitors:
            return "win" if selected == winner else "loss"
        return None
    away, home = _split_game(result.get("matchup"))
    if result.get("score_a") is not None and result.get("score_b") is not None:
        teams = (result.get("team_a"), result.get("team_b"))
        scores = (result["score_a"], result["score_b"])
    else:
        teams = (away, home)
        scores = (result.get("score_away"), result.get("score_home"))
    scores = tuple(number(score) for score in scores)
    if None in scores or not all(teams):
        return None
    if sport == "mlb":
        if market == "MONEYLINE" and result.get("moneyline_gradeable") is False:
            return None
        if market == "SPREAD" and result.get("spread_gradeable") is False:
            return None
        if market == "TOTAL" and not (result.get("total_gradeable") or (number(result.get("innings")) or 0) >= 9):
            return None
    if market == "TOTAL" and side["side"] in {"over", "under"} and side["line"] is not None:
        margin = (sum(scores) - side["line"]) * (1 if side["side"] == "over" else -1)
    else:
        selected = normalize_team_name(side["side"], sport)
        names = tuple(normalize_team_name(name, sport) for name in teams)
        if not selected or names.count(selected) != 1 or market == "TOTAL":
            return None
        index = names.index(selected)
        margin = scores[index] - scores[1 - index]
        if market == "SPREAD":
            if side["line"] is None:
                return None
            margin += side["line"]
    return "win" if margin > 0 else "loss" if margin < 0 else "push"


def candidates(db, sport, market, minutes, before, *, excluded_events=()):
    """Yield at most one priced, gradeable fixture per provider identity."""
    before = utc(before)
    used = set()
    windows = db.execute("""
        SELECT w.* FROM comparison_windows w
        WHERE (w.sport,w.market,w.minutes)=(?,?,?) AND w.decision_at<? AND w.scheduled_at<?
          AND NOT EXISTS (SELECT 1 FROM comparison_pending p WHERE
              (p.sport,p.event_id)=(w.sport,w.event_id))
        ORDER BY w.decision_at DESC,w.entry_at DESC,w.event_id
    """, (sport, market, minutes, before, before))
    for window in windows:
        if window["event_id"] in excluded_events:
            continue
        payload = decode(window["evidence"])
        outcomes = db.execute("""
            SELECT source,status,provider_event_id,observed_at,evidence FROM outcomes
            WHERE (sport,event_id)=(?,?) AND observed_at<=?
              AND status IN ('verified','push','void')
            ORDER BY observed_at DESC,digest
        """, (sport, window["event_id"], before))
        states = []
        identities = set()
        for source, status, identity, _, data in latest_provider_outcomes(outcomes, before):
            if not identity:
                continue
            result = json.loads(data)
            names = _split_game(result.get("matchup"))
            namespace = "nhl_league" if sport == "nhl" and source == "verified_research_nhl" else sport
            identity_key = namespace, identity
            identities.add(identity_key)
            # ESPN and the NHL league feed can use different IDs for one fixture.
            if sport == "nhl" and all(names):
                start = utc(result.get("provider_start") or result.get("fixture_scheduled_at"))
                if start:
                    matchup = tuple(sorted(normalize_team_name(name, sport) for name in names))
                    identities.add(("nhl_matchup", *matchup, start[:10]))
            states.append(tuple(
                grade(sport, market, side, status, result) if implied(side["price"]) is not None else None
                for side in payload["sides"]
            ))
        if not states or len(set(states)) != 1 or None in states[0] or used & identities:
            continue
        used.update(identities)
        yield {
            "event_id": window["event_id"], "decision_at": window["decision_at"],
            "entry_at": window["entry_at"], "scheduled_at": window["scheduled_at"],
            "sides": [value | {"result": state} for value, state in zip(payload["sides"], states[0])],
            "cross_entry_at": payload["cross_entry_at"], "published_evidence": payload["published_evidence"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=Path("/var/lib/redfox-history/private"))
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=480)
    parser.add_argument("--upgrade-existing", action="store_true")
    parser.add_argument("--upgrade-sport", choices=("nfl", "ncaaf", "nba", "ncaab", "nhl", "mlb", "ufc"))
    args = parser.parse_args()
    if shutil.disk_usage(args.state).free < FREE_DISK_RESERVE:
        print(packed({"deferred": "disk reserve"}))
        return
    with connect(args.state) as db:
        print(packed(rebuild(db, datetime.now(timezone.utc), args.limit,
                             upgrade_only=args.upgrade_existing or bool(args.upgrade_sport),
                             upgrade_sport=args.upgrade_sport, budget=args.seconds)))


if __name__ == "__main__":
    main()
