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


WINDOWS = (1440, 360, 120, 60, 40, 30, 20, 10)
MAX_AGE = timedelta(minutes=20)


def implied(price):
    if price is None or abs(price) < 100:
        return None
    return abs(price) / (100 + abs(price)) if price < 0 else 100 / (100 + price)


def delta(current, opening):
    return current - opening if current is not None and opening is not None else None


def features(history, entry, market):
    opening, current = implied(entry["open_price"]), implied(entry["price"])
    line_move = delta(entry["line"], entry["open_line"])
    direction = 1 if market == "TOTAL" and entry["side"] == "over" else -1
    if market == "MONEYLINE":
        path = [implied(row["price"]) for row in history]
        anchor = opening
    else:
        path = [direction * row["line"] for row in history]
        anchor = direction * entry["open_line"] if entry["open_line"] is not None else None
    path = [value for value in path if value is not None]
    steps = [1 if b > a else -1 for a, b in zip(path, path[1:]) if a != b]
    return {
        "side": entry["side"], "line": entry["line"], "price": entry["price"],
        "open_line": entry["open_line"], "open_price": entry["open_price"],
        "tickets": entry["tickets"], "money": entry["money"],
        "line_toward": None if line_move is None else line_move * direction,
        "price_toward_implied": delta(current, opening),
        "path_max_toward": max(path) - anchor if path and anchor is not None else None,
        "path_retrace": max(path) - path[-1] if path else None,
        "path_direction_changes": sum(a != b for a, b in zip(steps, steps[1:])),
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
                cross = "MONEYLINE" if market == "SPREAD" else "SPREAD" if market == "MONEYLINE" else None
                cross_at = None
                if cross:
                    _, related = latest(markets[cross], clocks[cross], deadline)
                    if related and related[0]["scheduled_at"] == schedule:
                        indexed = {row["side"]: row for row in related}
                        cross_at = related[0]["captured_at"]
                        for value in values:
                            other = indexed.get(value["side"])
                            if other:
                                value["cross_market"] = {
                                    "market": cross,
                                    "line_toward": delta(other["open_line"], other["line"]),
                                    "price_toward_implied": delta(implied(other["price"]), implied(other["open_price"])),
                                }
                context = db.execute("""
                    SELECT known_at,digest FROM decisions
                    WHERE (sport,event_id,market)=(?,?,?) AND source='published_board' AND known_at<=?
                    ORDER BY known_at DESC,digest LIMIT 1
                """, (sport, event, market, deadline)).fetchone()
                known_match[market, minutes] = (
                    sport, market, minutes, event, deadline, pair[0]["captured_at"], schedule,
                    encode({"sides": values, "cross_entry_at": cross_at, "verified_start": actual,
                            "published_evidence": dict(context) if context else None}),
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
            namespace = "nhl" if sport == "nhl" and source == "verified_research_nhl" else sport
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
