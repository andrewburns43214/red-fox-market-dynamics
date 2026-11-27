"""Build protected Historical Market Comparison payloads outside the board refresh."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3

from history_blocks import decode, encode, read_quotes
from history_comparisons import WINDOWS, cross_evidence, delta, features, grade, implied, recorded_split_context
from history_store import FREE_DISK_RESERVE, SPORTS, atomic_text, digest, eligible_pairs, number, packed, quote_part, start_evidence, utc
from team_aliases import _split_game, normalize_team_name
from ufc_bout_integrity import fighter_key


MARKETS = ("SPREAD", "MONEYLINE", "TOTAL")
KEY = re.compile(r"[A-Za-z0-9_-]{1,75}\Z")
MARKET_LABELS = {"SPREAD": "Spread", "MONEYLINE": "Moneyline", "TOTAL": "Total"}
CROSS = {"nfl": ("ncaaf",), "ncaaf": ("nfl",), "nba": ("ncaab",), "ncaab": ("nba",)}
KEY_NUMBERS = {"nfl": (3, 7, 10, 14), "ncaaf": (3, 7, 10, 14), "nhl": (1.5,), "mlb": (1.5,)}
MIN_HISTORY = 15
FROZEN_SCHEMA = """
CREATE TABLE IF NOT EXISTS comparison_frozen(
    sport TEXT NOT NULL, event_id TEXT NOT NULL, frozen_at TEXT NOT NULL, payload BLOB NOT NULL,
    PRIMARY KEY(sport,event_id)
) WITHOUT ROWID
"""


def side_key(value, sport, market):
    value = str(value or "").strip()
    if market == "TOTAL":
        label = value.split(" ", 1)[0].casefold()
        return {"under": "under", "u": "under", "over": "over", "o": "over"}.get(label, "")
    value = re.sub(r"\s[+-]?\d+(?:\.\d+)?(?:\s*\([^)]*\))?\s*$", "", value) if market == "SPREAD" else value
    return fighter_key(value) if sport == "ufc" else normalize_team_name(value, sport)


def line_price(value, market):
    if market == "MONEYLINE":
        n = number(value)
        return None, int(n) if n is not None and n.is_integer() and abs(n) >= 100 else None
    line, price, _ = quote_part("Market " + str(value or ""))
    return line, price if implied(price) is not None else None


def pct(value):
    n = number(value)
    return n if n is not None and 0 <= n <= 100 else None


def decision_window(minutes):
    # Compare snapshots at or before the closest available lead time. Cap
    # the mismatch so a next-week setup can't borrow a 24-hour comparison.
    best = min(WINDOWS, key=lambda horizon: abs(horizon - minutes))
    return best if abs(best - minutes) <= max(10, best * 0.28) else None


def metadata(value):
    return {key: value[key] for key in (
        "reaction", "path", "data_badge", "evidence_role", "whipsaw_recovered", "context_chips",
        "key_numbers_crossed", "response_direction", "active_worsening_reversal",
    ) if key in value}


def query_features(db, group, row, frozen, now):
    sport, event = row["sport"], row["game_id"]
    market = row.get("market_display")
    if market not in MARKETS or (sport == "ufc" and market != "MONEYLINE"):
        return None
    scheduled = utc(row.get("kickoff_iso"))
    observed = utc(row.get("source_latest_at"))
    if not scheduled or not observed or observed > utc(now) or observed >= scheduled:
        return None
    actual = start_evidence(db, sport, event)
    if actual and observed >= actual or frozen and sport == "ufc" and actual is None:
        return None
    if not frozen and (scheduled <= utc(now) or
                       now - datetime.fromisoformat(observed) > timedelta(minutes=35)):
        return None
    try:
        raw = json.loads(row.get("market_sides") or "[]")
    except ValueError:
        return None
    if not isinstance(raw, list) or not 1 <= len(raw) <= 2 or any(
        not isinstance(side, dict) or
        utc(side.get("source_latest_at")) != observed or
        str(side.get("data_badge") or row.get("data_badge") or "").upper() in {"RETAINED", "STALE", "INVALID"}
        for side in raw
    ):
        return None
    names = ("Over", "Under") if market == "TOTAL" else _split_game(row["game"])
    keyed = {side_key(value.get("flagged_side"), sport, market): value for value in raw}
    keys = [side_key(name, sport, market) for name in names]
    if not all(keys) or len(set(keys)) != 2 or len(keyed) != len(raw) or not set(keyed) <= set(keys):
        return None
    histories = defaultdict(list)
    for quote in read_quotes(db, (sport, event, market), until=observed):
        identity = side_key(quote["side"], sport, market)
        # Opening splits must come from a complete unambiguous snapshot.
        histories[quote["captured_at"]].append((identity, quote))
    selected = defaultdict(list)
    for captured, values in histories.items():
        if len(values) == 2 and {identity for identity, _ in values} == set(keys) and all(
                q["scheduled_at"] and captured < q["scheduled_at"] and (not actual or captured < actual)
                for _, q in values):
            for identity, quote in values:
                selected[identity].append(quote)
    paired_market = "MONEYLINE" if market == "SPREAD" else "SPREAD" if market == "MONEYLINE" else ""
    related = group.get(paired_market)
    cross = {}
    if related:
        cross_at = utc(related.get("source_latest_at"))
        if (cross_at and cross_at <= observed and utc(related.get("kickoff_iso")) == scheduled and
                datetime.fromisoformat(observed) - datetime.fromisoformat(cross_at) <= timedelta(minutes=20)):
            try:
                related_sides = json.loads(related.get("market_sides") or "[]")
            except ValueError:
                related_sides = []
            if not isinstance(related_sides, list):
                related_sides = []
            cross = {side_key(item.get("flagged_side"), sport, paired_market): item for item in related_sides
                     if isinstance(item, dict) and utc(item.get("source_latest_at")) == cross_at and
                     str(item.get("data_badge") or "").upper() in {"CLEAN", "THIN"}}
            if set(cross) != set(keys):
                cross = {}
    archived_cross = {}
    cross_priors = defaultdict(list)
    if related:
        cross_pairs = list(eligible_pairs(db, sport, event, paired_market, observed, actual_start=actual))
        if cross_pairs:
            cross_at, pair = cross_pairs[-1]
            indexed = {side_key(quote["side"], sport, paired_market): quote for quote in pair}
            if (set(indexed) == set(keys) and pair[0]["scheduled_at"] == scheduled and
                    datetime.fromisoformat(observed) - datetime.fromisoformat(cross_at) <= timedelta(minutes=20)):
                archived_cross = indexed
                for captured, prior_pair in cross_pairs[-3:-1]:
                    if (prior_pair[0]["scheduled_at"] == scheduled and
                            datetime.fromisoformat(cross_at) - datetime.fromisoformat(captured) <= timedelta(minutes=30)):
                        for quote in prior_pair:
                            cross_priors[side_key(quote["side"], sport, paired_market)].append(quote)
    values = []
    audits = recorded_split_context(db, sport, event, market, observed)
    for identity, name in zip(keys, names):
        raw_side = keyed.get(identity, {})
        if raw_side:
            line, price = line_price(raw_side.get("current_line"), market)
            opening, open_price = line_price(raw_side.get("open_line"), market)
            if price is None or market != "MONEYLINE" and line is None:
                return None
            entry = {"side": identity, "line": line, "price": price, "open_line": opening,
                     "open_price": open_price, "tickets": pct(raw_side.get("bets_pct")),
                     "money": pct(raw_side.get("money_pct")), "captured_at": observed}
        else:
            # Some frozen boards preserve only the Featured Side. Restore its
            # opposite only from the same verified paired capture and schedule.
            pair = {key: quotes[-1] for key, quotes in selected.items() if quotes and
                    quotes[-1]["captured_at"] == observed and quotes[-1]["scheduled_at"] == scheduled}
            if set(pair) != set(keys):
                return None
            recorded = next(iter(keyed))
            paired = pair[recorded]
            if line_price(keyed[recorded].get("current_line"), market) != (paired["line"], paired["price"]):
                return None
            entry = {**pair[identity], "side": identity}
        history = [quote for quote in selected[identity] if quote["captured_at"] < observed]
        value = features([*history, entry], entry, market)
        if not history:
            for name_of_value in ("opening_tickets", "opening_money", "ticket_change", "money_change",
                                  "path_max_toward", "path_min_toward", "path_retrace", "path_direction_changes",
                                  "duration_minutes", "tickets_after_move", "money_after_move"):
                value[name_of_value] = None
        value.update(name=name, context=metadata(raw_side), index=len(values), captured_at=observed)
        if identity in audits:
            value["split_audit"] = audits[identity]
        value["favorite"] = (str(row.get("red_fox_favorite")).casefold() == "true" and
                             side_key(row.get("favorite_side"), sport, market) == identity)
        if identity in cross:
            cross_line, cross_price = line_price(cross[identity].get("current_line"), paired_market)
            cross_open, cross_open_price = line_price(cross[identity].get("open_line"), paired_market)
            same_archive_capture = identity in archived_cross and utc(cross[identity]["source_latest_at"]) == cross_at
            quote = {"line": cross_line, "price": cross_price, "open_line": cross_open, "open_price": cross_open_price}
            value["cross_market"] = cross_evidence(paired_market, quote, cross_priors[identity] if same_archive_capture else ())
        elif identity in archived_cross:
            value["cross_market"] = cross_evidence(paired_market, archived_cross[identity], cross_priors[identity])
        values.append(value)
    lead = (datetime.fromisoformat(scheduled) - datetime.fromisoformat(observed)).total_seconds() / 60
    return {"observed": observed, "scheduled": scheduled, "lead": lead,
            "window": decision_window(lead), "sides": values}


def movement(value, market):
    return value.get("price_toward_implied" if market == "MONEYLINE" else "line_toward")


def polarity(value, threshold):
    return None if value is None else 1 if value >= threshold else -1 if value <= -threshold else 0


def confirmation(value):
    cross = value.get("cross_market") or {}
    if not cross:
        return None
    return (polarity(cross.get("price_toward_implied"), .005) if cross["market"] == "MONEYLINE"
            else polarity(cross.get("line_toward"), .25))


def distance(current, previous, sport, market, cross_sport=False):
    scale = .018 if market == "MONEYLINE" else (.7 if sport in {"nfl", "ncaaf", "nhl", "mlb"} else 1.5)
    left_move, right_move = movement(current, market), movement(previous, market)
    if any(current.get(k) is None or previous.get(k) is None for k in ("tickets", "money")):
        return None
    if left_move is None or right_move is None:
        return None
    if market == "TOTAL" and current["side"] != previous["side"]:
        return None
    if abs(current["tickets"] - previous["tickets"]) > 26 or abs(current["money"] - previous["money"]) > 30:
        return None
    if abs(left_move - right_move) > 3.1 * scale:
        return None
    factors = []

    def numeric(label, left, right, tolerance, weight):
        if left is not None and right is not None:
            factors.append((label, min(3, abs(left - right) / tolerance), weight))

    def flag(label, left, right, weight):
        if left is not None and right is not None:
            factors.append((label, 0 if left == right else 1.5, weight))

    numeric("Ticket share", current["tickets"], previous["tickets"], 9, 2)
    numeric("Money share", current["money"], previous["money"], 12, 2)
    numeric("Ticket–money divergence", delta(current["money"], current["tickets"]),
            delta(previous["money"], previous["tickets"]), 12, .8)
    numeric("Move size", left_move, right_move, scale, 2)
    numeric("Entry price", implied(current["price"]), implied(previous["price"]), .07, 1.15)
    numeric("Opening price", implied(current["open_price"]), implied(previous["open_price"]), .07, .65)
    numeric("Attached price movement", current.get("price_toward_implied"), previous.get("price_toward_implied"), .02, .75)
    if market != "MONEYLINE":
        line_scale = (4 if sport in {"nba", "ncaab", "nfl", "ncaaf"} else 1) if market == "TOTAL" else scale * 2.5
        numeric("Opening line", current.get("open_line"), previous.get("open_line"), line_scale, 1)
        numeric("Entry line", current.get("line"), previous.get("line"), line_scale, .5)
        if market == "SPREAD" and sport in KEY_NUMBERS:
            def crossings(side):
                opening, closing = side.get("open_line"), side.get("line")
                if opening is None or closing is None:
                    return None
                return tuple(n for n in KEY_NUMBERS[sport] if opening != closing and
                             any(min(opening, closing) <= v <= max(opening, closing) for v in (n, -n)))
            flag("Key-number movement", crossings(current), crossings(previous), .6)
    cutoff = .005 if market == "MONEYLINE" else .25
    same_direction = polarity(left_move, cutoff) == polarity(right_move, cutoff)
    flag("Move direction", polarity(left_move, cutoff), polarity(right_move, cutoff), 1.5)
    current_price, previous_price = implied(current["price"]), implied(previous["price"])
    flag("Favorite or underdog pricing", polarity(delta(current_price, .5), .003),
         polarity(delta(previous_price, .5), .003), .75)
    current_open, previous_open = implied(current["open_price"]), implied(previous["open_price"])
    if market == "MONEYLINE" and None not in (current_open, current_price, previous_open, previous_price):
        flag("Underdog-to-favorite move", current_open < .5 < current_price,
             previous_open < .5 < previous_price, .6)
    for label, key, tolerance, weight in (
        ("Opening tickets", "opening_tickets", 14, .55), ("Opening money", "opening_money", 16, .55),
        ("Ticket support change", "ticket_change", 12, .75), ("Money support change", "money_change", 15, .75),
        ("Tickets after earlier favorable movement", "tickets_after_move", 12, .6),
        ("Money after earlier favorable movement", "money_after_move", 15, .6),
        ("Retracement", "path_retrace", scale, .85), ("Earlier adverse move", "path_min_toward", scale, .6),
        ("Reversal count", "path_direction_changes", 2, .65),
    ):
        numeric(label, current.get(key), previous.get(key), tolerance, weight)
    if current.get("duration_minutes") and previous.get("duration_minutes"):
        numeric("Observation duration", math.log1p(current["duration_minutes"]),
                math.log1p(previous["duration_minutes"]), 1, .35)
    if current.get("observations") and previous.get("observations"):
        numeric("Observation count", math.log1p(current["observations"]),
                math.log1p(previous["observations"]), .8, .3)
    current_context = current.get("context") or {}
    previous_context = previous.get("context") or {}
    if not cross_sport:
        flag("Same-side cross-market direction", confirmation(current), confirmation(previous), 1.25)
        flag("Sustained cross-market support", (current.get("cross_market") or {}).get("sustained_toward"),
             (previous.get("cross_market") or {}).get("sustained_toward"), .65)
        flag("Recorded split breach pattern", (current.get("split_audit") or {}).get("split_breach"),
             (previous.get("split_audit") or {}).get("split_breach"), .35)
        for key, label, weight in (
            ("reaction", "Market Read", .65), ("path", "Market behavior", .6),
            ("evidence_role", "Pressure or resistance role", .5),
            ("data_badge", "Data quality", .45), ("whipsaw_recovered", "Whipsaw recovery", .4),
        ):
            flag(label, current_context.get(key), previous_context.get(key), weight)
    score = sum(d * w for _, d, w in factors) / sum(w for _, _, w in factors)
    tight = (same_direction and abs(current["tickets"] - previous["tickets"]) <= 10 and
             abs(current["money"] - previous["money"]) <= 13 and abs(left_move - right_move) <= scale and
             (previous_price is None or abs(current_price - previous_price) <= .09) and score <= .74)
    return score, tight, factors


def tournament_identity(sport, event, game, scheduled):
    names = _split_game(game)
    if not all(names) or sport == "mlb":
        return sport, event  # MLB day/name keys collapse doubleheaders.
    teams = tuple(sorted(side_key(name, sport, "MONEYLINE") for name in names))
    return sport, teams, scheduled[:10]


def score_line(result):
    if result.get("winner"):
        return "Winner: " + str(result["winner"])
    pairs = ((result.get("team_a"), result.get("score_a")), (result.get("team_b"), result.get("score_b")))
    if not all(name and score is not None for name, score in pairs):
        teams = _split_game(result.get("matchup"))
        pairs = ((teams[0], result.get("score_away")), (teams[1], result.get("score_home")))
    if all(name and number(score) is not None for name, score in pairs):
        return " · ".join(f"{name} {number(score):g}" for name, score in pairs)
    return None


class History:
    def __init__(self, db):
        self.db = db
        self.slots = {}
        self.results = {}
        self.cutoff = {}

    def windows(self, sport, market, minutes):
        key = sport, market, minutes
        if key not in self.slots:
            self.slots[key] = [dict(row) | decode(row["evidence"]) for row in self.db.execute("""
                SELECT w.*, f.matchup FROM comparison_windows w JOIN fixtures f USING(sport,event_id)
                WHERE (w.sport,w.market,w.minutes)=(?,?,?)
                  AND NOT EXISTS (SELECT 1 FROM comparison_pending p WHERE
                                  (p.sport,p.event_id)=(w.sport,w.event_id))
                ORDER BY w.decision_at DESC,w.event_id
            """, key)]
        return self.slots[key]

    def evidence(self, sport, event, before):
        key = sport, event
        if key not in self.results:
            self.results[key] = [(r["source"], r["status"], r["provider_event_id"], r["observed_at"], json.loads(r["evidence"]))
                                 for r in self.db.execute("""
                                     SELECT source,status,provider_event_id,observed_at,evidence
                                     FROM outcomes WHERE (sport,event_id)=(?,?)
                                 """, key)]
        return [r for r in self.results[key] if r[3] <= before]

    def rank(self, query, sport, event, market, window, before, cross=False):
        ranked = []
        sports = CROSS.get(sport, ()) if cross and market == "MONEYLINE" else () if cross else (sport,)
        for league in sports:
            for row in self.windows(league, market, window):
                if row["event_id"] == event and league == sport or row["scheduled_at"] >= before:
                    continue
                if row["entry_at"] > row["decision_at"] or row["decision_at"] >= before:
                    continue
                best = None
                for side in row["sides"]:
                    quality = distance(query, side, sport, market, cross)
                    if quality is not None:
                        option = quality, side
                        if best is None or (quality[0], side["side"]) < (best[0][0], best[1]["side"]):
                            best = option
                if best and best[0][0] <= 1.24:
                    quality, side = best
                    ranked.append((quality[0], row["event_id"], row, side, quality[1], quality[2]))
        ranked.sort(key=lambda item: (item[0], item[2]["sport"], item[1], item[3]["side"]))
        return ranked

    def report(self, ranked, market, before, examples=True):
        used = set()
        groups, grades, odds, numbers, exemplars = [], [], [], [], []
        missing_price, unresolved, void = 0, 0, 0
        for score, event, row, side, tight, factors in ranked:
            league = row["sport"]
            sources = self.evidence(league, event, before)
            identities = {tournament_identity(league, event, row["matchup"], row["scheduled_at"])}
            identities.update(("provider", "nhl-league" if s == "verified_research_nhl" else league, key)
                              for s, _, key, _, _ in sources if key)
            if used & identities:
                continue
            used.update(identities)
            groups.append(factors)
            p = implied(side["price"])
            if p is None:
                missing_price += 1
                continue
            results = [(grade(league, market, side, state, result), result) for _, state, key, _, result in sources if key]
            verified = {g for g, _ in results}
            if len(verified) != 1 or None in verified:
                unresolved += 1
                continue
            result = verified.pop()
            if result == "void":
                void += 1
                continue
            payout = 1 / p
            odds.append(payout)
            if side.get("line") is not None:
                numbers.append(side["line"])
            grades.append(result)
            if examples and len(exemplars) < 3:
                common = sorted((d, label) for label, d, _ in factors if d <= .75)
                differences = sorted(((d, label) for label, d, _ in factors if d > 1), reverse=True)
                exemplars.append({
                    "sport": league, "game": row["matchup"], "date": row["scheduled_at"],
                    "entry": row["entry_at"], "side": side, "grade": result,
                    "selection": next((name for name in _split_game(row["matchup"])
                                       if side_key(name, league, market) == side["side"]), side["side"].title()),
                    "score": next((score_line(e) for _, e in results if score_line(e)), None),
                    "matched": [label for _, label in common[:4]], "differences": [label for _, label in differences[:3]],
                })
        counts = {key: grades.count(key) for key in ("win", "loss", "push")}
        units = sum(o - 1 if g == "win" else -1 if g == "loss" else 0 for g, o in zip(grades, odds))
        decimal = sum(odds) / len(odds) if odds else None
        average = None if decimal is None else round(
            (decimal - 1) * 100 if decimal >= 2 else -100 / (decimal - 1)
        )
        labels = defaultdict(list)
        for factors in groups:
            for label, d, _ in factors:
                labels[label].append(d)
        matches, differences = [], []
        for label, values in sorted(labels.items(), key=lambda item: (sum(item[1]) / len(item[1]), item[0])):
            if len(values) < max(1, len(groups) * .5):
                continue
            agreement = sum(n <= .75 for n in values) / len(values)
            if agreement >= .65:
                matches.append(label)
            elif agreement < .50:
                differences.append(label)
        decisive = counts["win"] + counts["loss"]
        return {"matched": len(groups), "graded": len(grades), **counts, "void": void,
                "unresolved": unresolved, "missing_price": missing_price,
                "win_rate": round(100 * counts["win"] / decisive, 1) if decisive else None,
                "odds": average, "line": round(sum(numbers) / len(numbers), 2) if numbers else None,
                "units": round(units, 2) if grades else None,
                "roi": round(units * 100 / len(grades), 2) if grades else None,
                "matches": matches[:8], "differences": differences[:5], "examples": exemplars,
                "status": "none" if not groups else "limited" if len(grades) < MIN_HISTORY else "available"}

    def comparison(self, query, sport, event, market, setup):
        window, before = setup["window"], setup["observed"]
        if window is None:
            return {"reason": "There is no historical checkpoint close to this observation time."}
        ranked = self.rank(query, sport, event, market, window, before)
        closest = self.report([r for r in ranked if r[4]], market, before)
        broader = self.report(ranked, market, before)
        since = min((r["scheduled_at"][:10] for r in self.windows(sport, market, window)
                     if r["scheduled_at"] < before), default=None)
        related = self.rank(query, sport, event, market, window, before, cross=True)
        cross = self.report(related, market, before, examples=False) if related else None
        return {"closest": closest, "broader": broader, "cross": cross, "since": since,
                "cross_sports": sorted({item[2]["sport"] for item in related})}


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fixture_groups(rows):
    groups = {}
    for row in rows:
        sport, game_id, market = row.get("sport"), row.get("game_id"), row.get("market_display")
        if sport in SPORTS and KEY.fullmatch(game_id or "") and market in MARKETS:
            group = groups.setdefault((sport, game_id), {})
            if market not in group or (utc(row.get("source_latest_at")) or "") > (
                    utc(group[market].get("source_latest_at")) or ""):
                group[market] = row
    return groups


def render_fixture(db, history, key, group, now, frozen):
    sport, event = key
    markets = {}
    for market, row in group.items():
        setup = query_features(db, group, row, frozen, now)
        if setup:
            for query in setup["sides"]:
                query["comparison"] = history.comparison(query, sport, event, market, setup)
            markets[market] = setup
    return {"sport": sport, "event": event, "game": next(iter(group.values()))["game"],
            "markets": markets, "frozen": frozen}


def write(path, content):
    with atomic_text(path, packed(content)):
        pass
    path.chmod(0o644)


def build(db, root, output, control, now, *, preview=False, events=()):
    if not preview:
        db.execute(FROZEN_SCHEMA)
    output.mkdir(parents=True, exist_ok=True, mode=0o755)
    manifest = output / "index.json"
    if control.exists():
        write(manifest, {"enabled": False, "games": [], "generated": utc(now)})
        return {"disabled": True}
    if shutil.disk_usage(output).free < FREE_DISK_RESERVE:
        return {"deferred": "disk reserve"}
    board = fixture_groups(read_rows(root / "data/anomaly_board.csv"))
    recent = fixture_groups(row for row in read_rows(root / "data/live_recent.csv")
                            if utc(row.get("frozen_at_utc")))
    history = History(db)
    stamp = utc(now)
    # Global token is cheap and changes whenever historical materialization
    # or verified outcomes change. Keep current quote signatures per event.
    revision = tuple(db.execute("""
        SELECT (SELECT count(*) FROM comparison_windows), (SELECT max(observed_at) FROM outcomes),
        (SELECT value FROM jobs WHERE name='comparison_generation')
    """).fetchone())
    keys = []
    available = {}
    updated, frozen_count = 0, 0
    for key in dict.fromkeys((*recent, *board)):
        if events and key[1] not in events:
            continue
        group = recent.get(key) or board[key]
        frozen = key in recent
        path = output / f"{key[0]}--{key[1]}.json"
        signature = digest(packed((group, revision)))
        if frozen and not preview and (old := db.execute("SELECT payload FROM comparison_frozen WHERE (sport,event_id)=(?,?)",
                                         key).fetchone()):
            if not path.exists():
                write(path, decode(old[0]))
            else:
                visible = json.loads(path.read_text(encoding="utf-8"))
                if not visible.get("frozen"):
                    write(path, decode(old[0]))
            keys.append(path.stem)
            available[path.stem] = list(decode(old[0])["markets"])
            continue
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if old.get("signature") == signature and not frozen:
            valid = [market for market, item in old["markets"].items()
                     if now - datetime.fromisoformat(item["observed"]) <= timedelta(minutes=35)
                     and item["scheduled"] > stamp]
            if valid:
                keys.append(path.stem)
                available[path.stem] = valid
            continue
        payload = render_fixture(db, history, key, group, now, frozen)
        if not payload["markets"]:
            continue
        payload.update(signature=signature, generated=stamp)
        if frozen and not preview:
            with db:
                db.execute("INSERT OR IGNORE INTO comparison_frozen VALUES (?,?,?,?)",
                           (*key, min(row["frozen_at_utc"] for row in group.values()), encode(payload)))
            frozen_count += 1
        write(path, payload)
        keys.append(path.stem)
        available[path.stem] = list(payload["markets"])
        updated += 1
    write(manifest, {"enabled": True, "games": available, "generated": stamp})
    present = set(keys)
    for path in output.glob("*--*.json"):
        if path.stem not in present and now.timestamp() - path.stat().st_mtime > 86400:
            path.unlink()
    return {"games": len(keys), "updated": updated, "frozen": frozen_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/opt/red-fox-market-dynamics"))
    parser.add_argument("--state", type=Path, default=Path("/var/lib/redfox-history/private"))
    parser.add_argument("--output", type=Path, default=Path("/var/lib/redfox-history/public/comparisons"))
    parser.add_argument("--disable-file", type=Path, default=Path("/etc/redfox-history-comparisons.disabled"))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--event", action="append", default=[])
    args = parser.parse_args()
    if args.preview and args.output.resolve() == Path("/var/lib/redfox-history/public/comparisons"):
        parser.error("a preview needs an isolated --output directory")
    db_path = args.state / "history.sqlite3"
    connection = db_path.resolve().as_uri() + "?mode=ro" if args.preview else db_path
    with sqlite3.connect(connection, timeout=20, uri=args.preview) as db:
        db.row_factory = sqlite3.Row
        print(packed(build(db, args.root, args.output, args.disable_file, datetime.now(timezone.utc),
                           preview=args.preview, events=args.event)))


if __name__ == "__main__":
    main()
