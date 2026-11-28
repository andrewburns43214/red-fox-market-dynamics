"""Descriptive, outcome-independent rules for presenting a verified comparison ledger."""

from bisect import bisect_right

from history_comparisons import implied


VERSION = 1
FOOTBALL_KEYS = (3, 4, 6, 7, 10, 14)


def key_path(quote, keys):
    opening, observed = quote.get("open_line"), quote.get("line")
    if opening is None or observed is None or opening == observed:
        return ()
    return tuple((number, "off" if opening == number else "onto" if observed == number else "through")
                 for key in keys for number in (-key, key)
                 if min(opening, observed) <= number <= max(opening, observed))


def direction(market, quote):
    moneyline = market == "MONEYLINE"
    value = quote.get("price_toward_implied" if moneyline else "line_toward")
    minimum = .005 if moneyline else .25
    return None if value is None else 0 if abs(value) < minimum else (1 if value > 0 else -1)


def total_direction(quote):
    opening, line = quote.get("open_line"), quote.get("line")
    return None if opening is None or line is None else (line > opening) - (line < opening)


def role(market, quote, opposite):
    if market == "TOTAL":
        return ()
    roles = []
    for field in ("open_" + ("price" if market == "MONEYLINE" else "line"),
                  "price" if market == "MONEYLINE" else "line"):
        value = quote.get(field)
        if market == "MONEYLINE":
            ours, theirs = implied(value), implied(opposite.get(field))
            value = theirs - ours if ours is not None and theirs is not None else None
        roles.append(None if value is None else (value < 0) - (value > 0))
    return tuple(roles)


def reliable_splits(quote, opposite):
    """Complementary extreme splits can be reported, but a joint cap cannot establish similarity."""
    for field in ("tickets", "money"):
        a, b = quote.get(field), opposite.get(field)
        if a is None or b is None or not (0 <= a <= 100 and 0 <= b <= 100 and 99 <= a + b <= 101):
            return False
    return (quote["tickets"], quote["money"]) not in ((0, 0), (100, 100))


def opener(market, quote):
    return implied(quote.get("open_price")) if market == "MONEYLINE" else quote.get("open_line")


def same_opener(sport, market, a, b):
    if a is None or b is None:
        return False
    if market == "MONEYLINE":
        return abs(a - b) <= .05
    if market == "TOTAL":
        return abs(a - b) <= (.75 if sport in {"mlb", "nhl"} else 4)
    # Use both an absolute gap and a band that respects the different scale
    # of college lines. Football endpoint behavior is scored separately.
    if abs(a - b) > (1 if sport in {"mlb", "nhl"} else 2.5 if sport in {"nba", "ncaab"} else 2):
        return False
    bands = (3, 7, 14, 21) if sport == "ncaaf" else ()
    return not bands or bisect_right(bands, abs(a)) == bisect_right(bands, abs(b))


def partitions(ledger, reports, sport, market, side, opposite):
    """Partition the candidate pool and attach at most three setup-driven scenarios."""
    primary_role = role(market, side, opposite)
    primary_move = direction(market, side)
    current_open = opener(market, side)
    current_price = implied(side.get("price"))
    peers = side.get("game_context") or {}
    peer_market = "MONEYLINE" if market == "SPREAD" else "SPREAD"
    peer_move = direction(peer_market, peers.get(peer_market, {}))
    team_markets = ("SPREAD", "MONEYLINE")
    side_markets_held = market == "TOTAL" and all(direction(m, peers.get(m, {})) == 0 for m in team_markets)
    total_move = total_direction(peers.get("TOTAL", {}))
    keys = FOOTBALL_KEYS if sport in {"nfl", "ncaaf"} and market == "SPREAD" else ()
    keys_now = key_path(side, keys)
    trustworthy = reliable_splits(side, opposite)
    scenarios = {}
    best, related = [], []
    closest = set(ledger.groups["closest"])
    for index in ledger.groups["broader"]:
        row = ledger.rows[index]
        q, other = row["side"], row["opposite"]
        if not trustworthy or not reliable_splits(q, other):
            continue
        old_role = role(market, q, other)
        old_move = direction(market, q)
        old_peers = row["peers"]
        old_peer = direction(peer_market, old_peers.get(peer_market, {}))
        old_keys = key_path(q, keys)
        same_role_move = None not in primary_role and old_role == primary_role and primary_move is not None and old_move == primary_move
        comparable_line = same_opener(sport, market, current_open, opener(market, q))
        quote_price = implied(q.get("price"))
        comparable_price = None not in (current_price, quote_price) and abs(current_price - quote_price) <= .075
        peer_conflicts = market != "TOTAL" and peer_move is not None and old_peer is not None and old_peer != peer_move
        if index in closest and same_role_move and comparable_line and comparable_price and keys_now == old_keys and not peer_conflicts:
            best.append(index)
        else:
            related.append(index)

        def add(key, label):
            scenarios.setdefault(key, (label, []))[1].append(index)

        if not same_role_move:
            continue
        if keys_now and old_keys == keys_now:
            steps = {"off": "Off", "onto": "Onto", "through": "Through"}
            label = ", ".join(f"{steps[step]} {line:+g}" for line, step in keys_now)
            if side.get("trajectory") == "retraced":
                if q.get("trajectory") == "retraced":
                    add("key", label + " with buyback")
            else:
                add("key", label)
        if primary_role in ((-1, 1), (1, -1)):
            add("crossover", "Underdog became the favorite" if primary_role[0] == -1 else "Favorite became the underdog")
        if market != "TOTAL" and primary_move and peer_move is not None and old_peer == peer_move:
            if peer_move:
                add("confirmation", "Spread and moneyline moved together" if peer_move == primary_move else
                    "Spread and moneyline moved in opposite directions")
            else:
                add("confirmation", "Moneyline had little movement" if peer_market == "MONEYLINE" else "Spread handicap held")
        if primary_move and market != "TOTAL" and total_move and total_direction(old_peers.get("TOTAL", {})) == total_move:
            add("total", "Side moved with a rising total" if total_move > 0 else "Side moved with a falling total")
        if primary_move and side_markets_held and all(direction(m, old_peers.get(m, {})) == 0 for m in team_markets):
            add("total", "Total moved while spread and moneyline held")
        if side.get("trajectory") in {"retraced", "recovered"} and q.get("trajectory") == side["trajectory"]:
            add("trajectory", "Move followed by buyback" if side["trajectory"] == "retraced" else "Move recovered after a reversal")
        if market != "MONEYLINE" and primary_move == 0:
            price_move = direction("MONEYLINE", side)
            if price_move is not None and direction("MONEYLINE", q) == price_move:
                add("juice", "Same line, moving price" if price_move else "Little or no line or price movement")
        for minimum, maximum, description in ((65, 100, "Heavy public support"), (0, 35, "Low public support")):
            if all(minimum <= quote[field] <= maximum for quote in (side, q) for field in ("tickets", "money")):
                suffix = (" with no material move" if primary_move == 0 else
                          " despite movement the other way" if primary_move == -1 else " with movement toward it")
                add("pressure", description + suffix)
                break

    reports["best"] = ledger.add_population("best", best)
    reports["related"] = ledger.add_population("related", related)
    reports["comparison_version"] = VERSION
    reports["best"]["reason"] = None if trustworthy else "Paired betting splits could not be verified for this capture."
    reports["scenarios"] = []
    for key in ("key", "crossover", "confirmation", "juice", "trajectory", "total", "pressure"):
        if key not in scenarios:
            continue
        label, members = scenarios[key]
        group = "scenario_" + key
        reports[group] = ledger.add_population(group, members)
        reports["scenarios"].append({"key": group, "label": label})
        if len(reports["scenarios"]) == 3:
            break
