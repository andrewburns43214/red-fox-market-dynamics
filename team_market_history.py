"""Verified supplemental results and quotes for the Team Intelligence history tab."""

from collections import defaultdict
import csv
from datetime import date, datetime, timedelta
import math
from zoneinfo import ZoneInfo

from history_comparisons import grade, side_key

ET = ZoneInfo("America/New_York")
GRADES = {"win", "loss", "push"}


def quote_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def american_price(value):
    price = quote_number(value)
    return int(price) if price is not None and price.is_integer() and abs(price) >= 100 else None


def flat_units(result, price):
    if result not in GRADES or price is None:
        return None
    return round(100 / abs(price) if price < 0 else price / 100, 6) if result == "win" else -1 if result == "loss" else 0


def reference_quotes(sport, game, source, *, spread=(), total=(), money=(), spread_price=(), total_price=(),
                     open_money=(), open_spread_price=(), open_total_price=()):
    """Normalize a provider's full-game quote without inventing missing odds."""
    keys = game["keys"]
    away, home = game["scores"]
    receipt = {"matchup": " @ ".join(game["names"]), "score_away": away, "score_home": home}
    matrix = {}
    for market, sides, quotes, prices, open_prices in (
        ("SPREAD", keys, spread, spread_price, open_spread_price),
        ("TOTAL", ("over", "under"), total, total_price, open_total_price),
        ("MONEYLINE", keys, (), money, open_money),
    ):
        for i, key in enumerate(sides):
            opened, line = quotes[i] if quotes else (None, None)
            opened, line = quote_number(opened), quote_number(line)
            price = american_price(prices[i]) if prices else None
            open_price = american_price(open_prices[i]) if open_prices else None
            if line is None and price is None and opened is None and open_price is None:
                continue
            result = grade(sport, market, {"side": key, "line": line}, "verified", receipt)
            matrix.setdefault(market, {})[key] = {
                "line": line, "open_line": opened, "price": price, "grade": result,
                "open_price": open_price, "units": flat_units(result, price), "source": source,
            }
    return matrix


def game_date(game):
    return game.get("date") or datetime.fromisoformat(game["start"]).astimezone(ET).date().isoformat()


def match_key(game):
    return game_date(game), tuple(sorted(game["keys"]))


def score_key(game):
    if game.get("scores") is not None:
        return tuple(sorted(zip(game["keys"], game["scores"])))
    if game.get("winner"):
        return game["winner"]
    return game.get("result")


def timestamp(value):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp if stamp.tzinfo else None
    except (AttributeError, TypeError, ValueError):
        return None


def sportsbook_history(path, sport, archive, history, guard):
    """Recover paired pregame quotes already held in the existing sportsbook archive."""
    if not path.is_file():
        return
    candidates = defaultdict(list)
    for game in [*archive, *history["games"]]:
        kickoff = timestamp(game.get("market_start") or game.get("start"))
        if kickoff and not game.get("date"):
            candidates[(tuple(game["keys"]), kickoff)].append(game)
    if not candidates:
        return
    snapshots = defaultdict(dict)
    for_game = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for count, row in enumerate(csv.DictReader(stream)):
            if count % 4096 == 0 and not guard():
                raise TimeoutError("publisher priority")
            if row.get("sport") != sport:
                continue
            market, book = row.get("market"), row.get("bookmaker")
            if market not in {"MONEYLINE", "SPREAD", "TOTAL"} or not book or sport == "ufc" and market != "MONEYLINE":
                continue
            kickoff, observed = timestamp(row.get("commence_time")), timestamp(row.get("timestamp"))
            if not kickoff or not observed or observed >= kickoff:
                continue
            keys = tuple(side_key(row.get(field) or "", sport, "MONEYLINE")
                         for field in ("away_team_norm", "home_team_norm"))
            matched = candidates.get((keys, kickoff), [])
            if len(matched) != 1:
                continue
            game = matched[0]
            if market in game.get("quotes", {}) or market in game.get("history_quotes", {}):
                continue
            if game.get("actual") and observed >= timestamp(game["actual"]):
                continue
            changed = timestamp(row.get("changed_at"))
            if changed and changed > observed:
                continue
            expected = ("over", "under") if market == "TOTAL" else keys
            side = (row.get("side") or "").lower() if market == "TOTAL" else side_key(row.get("side") or "", sport, "MONEYLINE")
            line, odds = quote_number(row.get("line")), american_price(row.get("odds_american"))
            if side not in expected or odds is None or market != "MONEYLINE" and line is None:
                continue
            group = game["uid"], market, book, observed
            point = {"line": line if market != "MONEYLINE" else None, "price": odds}
            pair = snapshots[group]
            if side in pair and pair[side] != point:
                pair[side] = None
            else:
                pair[side] = point
            for_game[game["uid"]] = game
    verified = defaultdict(list)
    for (uid, market, book, stamp), pair in snapshots.items():
        if len(pair) != 2 or not all(pair.values()):
            continue
        lines = [point["line"] for point in pair.values()]
        if (market == "SPREAD" and lines[0] != -lines[1]) or (market == "TOTAL" and lines[0] != lines[1]):
            continue
        verified[uid, market].append((stamp, book, pair))
    for (uid, market), choices in verified.items():
        game = for_game[uid]
        stamp, book, pair = max(choices, key=lambda point: (point[0], point[1]))
        first, _, opened = min((point for point in choices if point[1] == book), key=lambda point: point[0])
        raw = game.get("_receipt")
        if not raw and game.get("scores"):
            raw = ("verified", {"matchup": " @ ".join(game["names"]), "score_away": game["scores"][0],
                                "score_home": game["scores"][1]})
        if not raw and sport == "ufc" and game.get("winner"):
            raw = ("verified", {"matchup": " vs ".join(game["names"]), "winner": game["winner"]})
        if not raw:
            continue
        patch = history["quotes"].setdefault(uid, {}).setdefault(market, {})
        for side, point in pair.items():
            result = grade(sport, market, {"side": side, "line": point["line"]}, *raw)
            if side not in patch or patch[side].get("units") is None and result in GRADES:
                patch[side] = {**point, "grade": result, "units": flat_units(result, point["price"]),
                               "source": book + " via The Odds API",
                               "open_line": opened[side]["line"] if first < stamp else None,
                               "open_price": opened[side]["price"] if first < stamp else None}


def merge_market_history(archive, layers, *, sport, blocked=()):
    """Join scores by verified identity and finals. Archive evidence takes priority."""
    by_match = defaultdict(list)
    for game in archive:
        by_match[match_key(game)].append(game)
    withheld = {match_key(game) for game in blocked}
    extra = []
    patches = {}
    for layer in layers:
        by_external_match = defaultdict(list)
        for game in layer:
            key = match_key(game)
            if sport == "ufc" and key not in by_match:
                local_date = date.fromisoformat(key[0])
                neighbours = [((local_date + timedelta(days=offset)).isoformat(), key[1]) for offset in (-1,1)]
                paired = [candidate for candidate in neighbours if any(
                    score_key(row) == score_key(game) for row in by_match.get(candidate, []))]
                if len(paired) == 1:
                    key = paired[0]
            by_external_match[key].append(game)
        for key, incoming in by_external_match.items():
            if key in withheld:
                continue
            existing = by_match[key]
            if sport != "mlb" and len(incoming) > 1:
                if len({score_key(game) for game in incoming}) != 1:
                    continue
                incoming = sorted(incoming, key=lambda game: not bool(game.get("quotes")))[:1]
            matched = defaultdict(list)
            for game in existing:
                matched[score_key(game)].append(game)
            for game in incoming:
                matches = matched.get(score_key(game), [])
                if (len(matches) == 1 and sum(score_key(row) == score_key(game) for row in incoming) == 1
                        and game.get("quotes")):
                    target = matches[0]
                    patch = patches.setdefault(target["uid"], {})
                    for market, sides in game["quotes"].items():
                        for side, quote in sides.items():
                            patch.setdefault(market, {}).setdefault(side, quote)
            # A same-day mismatch is a conflict unless the source accounts for the
            # archived games and shows an additional, separately identified game.
            counts = defaultdict(int)
            for game in incoming:
                counts[score_key(game)] += 1
            if any(len(matches) > counts[key] for key, matches in matched.items()):
                continue
            for game in incoming:
                matches = matched[score_key(game)]
                if matches:
                    matches.pop()
                    continue
                if game.get("reference_only"):
                    continue
                extra.append(game)
                existing.append(game)
    return {"games": extra, "quotes": patches}
