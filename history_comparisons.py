"""Materialize auditable historical decision windows for private research."""

import argparse
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil

from history_blocks import decode, encode
from history_store import FREE_DISK_RESERVE, connect, eligible_pairs, number, packed, start_evidence, utc
from team_aliases import _split_game, normalize_team_name
from ufc_bout_integrity import fighter_key


WINDOWS = (4320, 2880, 1440, 360, 120, 60, 40, 30, 20, 10)
MAX_AGE = timedelta(minutes=20)


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
    turn = next((row for row, value in zip(history, path) if anchor is not None and value is not None and
                 value - anchor >= (.005 if market == "MONEYLINE" else .25)), {})
    path = [value for value in path if value is not None]
    steps = [1 if b > a else -1 for a, b in zip(path, path[1:]) if a != b]
    first = history[0] if history else {}
    initial_tickets, initial_money = first.get("tickets"), first.get("money")
    start, end = utc(first.get("captured_at")), utc(entry.get("captured_at"))
    return {
        "side": entry["side"], "line": entry["line"], "price": entry["price"],
        "open_line": entry["open_line"], "open_price": entry["open_price"],
        "tickets": entry["tickets"], "money": entry["money"],
        "line_toward": None if line_move is None else line_move * direction,
        "price_toward_implied": delta(current, opening),
        "path_max_toward": max(path) - anchor if path and anchor is not None else None,
        "path_min_toward": min(path) - anchor if path and anchor is not None else None,
        "path_retrace": max(path) - path[-1] if path else None,
        "path_direction_changes": sum(a != b for a, b in zip(steps, steps[1:])),
        "opening_tickets": initial_tickets, "opening_money": initial_money,
        "ticket_change": delta(entry["tickets"], initial_tickets),
        "money_change": delta(entry["money"], initial_money),
        "tickets_after_move": delta(entry["tickets"], turn.get("tickets")),
        "money_after_move": delta(entry["money"], turn.get("money")),
        "split_gap": delta(entry["money"], entry["tickets"]),
        "observations": len(history),
        "duration_minutes": (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60
        if start and end else None,
    }


def latest(pairs, timestamps, deadline):
    at = bisect_right(timestamps, deadline) - 1
    if at >= 0 and datetime.fromisoformat(deadline) - datetime.fromisoformat(timestamps[at]) <= MAX_AGE:
        return at, pairs[at][1]
    return None, None


def generate(db, sport, event):
    actual = start_evidence(db, sport, event)
    markets = {
        market: list(eligible_pairs(db, sport, event, market, "9999", actual_start=actual))
        for market in ("SPREAD", "MONEYLINE", "TOTAL")
    }
    clocks = {market: [capture for capture, _ in pairs] for market, pairs in markets.items()}
    known_match = {}
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
                history = {row["side"]: [] for row in pair}
                for _, prior in pairs[:index + 1]:
                    for row in prior:
                        if row["side"] in history:
                            history[row["side"]].append(row)
                values = [features(history[row["side"]], row, market) for row in pair]
                audits = recorded_split_context(db, sport, event, market, deadline)
                for value in values:
                    name = value["side"]
                    identity = fighter_key(name) if sport == "ufc" else normalize_team_name(name, sport)
                    if identity in audits:
                        value["split_audit"] = audits[identity]
                cross = "MONEYLINE" if market == "SPREAD" else "SPREAD" if market == "MONEYLINE" else None
                cross_at = None
                if cross:
                    cross_index, related = latest(markets[cross], clocks[cross], deadline)
                    if related and related[0]["scheduled_at"] == schedule:
                        indexed = {row["side"]: row for row in related}
                        cross_at = related[0]["captured_at"]
                        for value in values:
                            other = indexed.get(value["side"])
                            if other:
                                prior = [row for captured, prior_pair in markets[cross][max(0, cross_index - 2):cross_index]
                                         if datetime.fromisoformat(cross_at) - datetime.fromisoformat(captured) <=
                                         timedelta(minutes=30) for row in prior_pair if row["side"] == value["side"]
                                         and row["scheduled_at"] == schedule]
                                value["cross_market"] = cross_evidence(cross, other, prior)
                context = db.execute("""
                    SELECT known_at,digest,evidence FROM decisions
                    WHERE (sport,event_id,market)=(?,?,?) AND source='published_board' AND known_at<=?
                    ORDER BY known_at DESC,digest LIMIT 1
                """, (sport, event, market, deadline)).fetchone()
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
                            if context else None, "version": 3}),
                )
                break
    return known_match.values()


def rebuild(db, now, limit):
    events = db.execute("""
        SELECT f.sport,f.event_id FROM comparison_pending p JOIN fixtures f USING(sport,event_id)
        WHERE f.scheduled_at<=? OR (f.scheduled_at IS NULL AND f.last_at<=?)
        ORDER BY f.last_at DESC LIMIT ?
    """, (utc(now - timedelta(hours=3)), utc(now - timedelta(days=1)), limit)).fetchall()
    windows = 0
    for sport, event in events:
        generated = generate(db, sport, event)
        with db:
            db.execute("DELETE FROM comparison_windows WHERE (sport,event_id)=(?,?)", (sport, event))
            db.executemany("INSERT INTO comparison_windows VALUES (?,?,?,?,?,?,?,?)", generated)
            db.execute("DELETE FROM comparison_pending WHERE (sport,event_id)=(?,?)", (sport, event))
        windows += len(generated)
    if events:
        with db:
            db.execute("INSERT OR REPLACE INTO jobs VALUES ('comparison_generation',?)", (utc(now),))
    return {"fixtures": len(events), "windows": windows}


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
            SELECT source,status,provider_event_id,evidence FROM outcomes
            WHERE (sport,event_id)=(?,?) AND observed_at<=?
              AND status IN ('verified','push','void')
        """, (sport, window["event_id"], before))
        states = []
        identities = set()
        for source, status, identity, data in outcomes:
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
    args = parser.parse_args()
    if shutil.disk_usage(args.state).free < FREE_DISK_RESERVE:
        print(packed({"deferred": "disk reserve"}))
        return
    with connect(args.state) as db:
        print(packed(rebuild(db, datetime.now(timezone.utc), args.limit)))


if __name__ == "__main__":
    main()
