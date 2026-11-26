"""NHL paper qualification predicates. This module does not publish Favorites."""

import pandas as pd

import red_fox_favorite as favorite

PUBLIC = "public_backed_moneyline"
PUBLIC_MIN_TICKETS = 80
PUBLIC_MIN_MONEY = 55
CROSSOVER = "genuine_crossover"
CROSSOVER_LOW = "genuine_crossover_contrarian_splits"
LANES = (PUBLIC, CROSSOVER, CROSSOVER_LOW)
GOALIE_RISKS = {"Goalie Change", "Goalie Unconfirmed", "Goalie Risk"}


def fair(price, other):
    if not price or not other or min(abs(price), abs(other)) < 100:
        return None
    a, b = (favorite._american_implied_probability(x) for x in (price, other))
    return a / (a + b)


def profile(lane, side, other, opener, current):
    """Evaluate only the lane's own signal, with original price boundaries."""
    odds = favorite._line_value(side.get("current_line"), "MONEYLINE")
    if odds is None or abs(odds) < 100:
        return "Invalid selected American odds"
    if lane == PUBLIC:
        if not -165 <= odds <= 120:
            return "Price outside −165/+120"
    elif not -165 <= odds <= -100:
        return "Crossover price outside −165/−100"

    values = [favorite._number(item.get(k)) for item in (side, other) for k in ("bets_pct", "money_pct")]
    if (any(v is None or not 0 < v < 100 for v in values)
            or abs(values[0] + values[2] - 100) > 2 or abs(values[1] + values[3] - 100) > 2):
        return "Invalid, capped or noncomplementary paired splits"
    bets, money, opp_bets, opp_money = values
    if lane == PUBLIC and (bets < PUBLIC_MIN_TICKETS or money < PUBLIC_MIN_MONEY):
        return "Public tickets below 80% or handle below 55%"
    if lane == CROSSOVER_LOW and not (bets <= 45 and money <= 45 and bets < opp_bets and money < opp_money):
        return "Existing Contrarian split restriction failed"
    if lane != PUBLIC and (opener is None or current is None or not opener < .5 < current):
        return "Genuine paired crossover no longer present"
    return ""


def assess(row, selected, lane, at, opener, context=None):
    """Return the first failed gate and whether it is an immediate safety veto."""
    if row.get("sport") != "nhl" or row.get("market_display") != "MONEYLINE":
        return "No observed paired NHL moneyline", False
    sides = favorite._sides(row)
    name = favorite._side_identity(selected)
    matching = [s for s in sides if favorite._side_identity(s.get("flagged_side")) == name]
    if len(sides) != 2 or len(matching) != 1:
        return "Market side identity unavailable or ambiguous", True
    side = matching[0]
    other = next(s for s in sides if s is not side)
    if (favorite._truthy(row.get("cross_market_mismatch")) or
            favorite._truthy(row.get("cross_market_opener_mismatch_verified"))):
        return "Verified cross-market contradiction", True
    if context and context.get("status") == "verified_change":
        return "Verified confirmed goalie pair changed", True
    for s in sides:
        source = pd.to_datetime(s.get("source_latest_at"), errors="coerce", utc=True)
        if pd.isna(source) or not pd.Timedelta(0) <= at-source <= pd.Timedelta(minutes=12):
            return "Paired source missing or older than 12 minutes", True
    chips = favorite._parts(side.get("context_chips")) | favorite._parts(side.get("anomaly_chips"))
    if chips & GOALIE_RISKS:
        return "Explicit goalie risk: "+", ".join(sorted(chips & GOALIE_RISKS)), True
    issue = favorite._quality_failure(side)
    if issue:
        return issue, True
    recovered = favorite._truthy(side.get("whipsaw_recovered")) or "Whipsaw Recovered" in chips
    if recovered or side.get("path") == "Whipsaw":
        if (not recovered or (favorite._number(side.get("whipsaw_retention_ratio")) or 0) < .8
                or (favorite._number(side.get("material_direction_changes")) or 0) > 2):
            return "Whipsaw recovery, 80% retention or reversal limit failed", True
    # Empty or descriptive Supported Side is not a requirement. A separately
    # supported opposite side or verified opposite directional action is.
    support = favorite._side_identity(row.get("supported_side"))
    opposite = favorite._side_identity(other.get("flagged_side"))
    opposite_action = other.get("reaction") in ("Contrarian", "Follow") and favorite._truthy(other.get("kpi_eligible"))
    if support == opposite or opposite_action:
        return "Confirmed opposite directional signal", True
    a, b = (favorite._line_value(s.get("current_line"), "MONEYLINE") for s in (side, other))
    return profile(lane, side, other, opener, fair(a, b)), False


def advance(state, at, remaining, quote, issue, hard):
    """Advance at one distinct paired source timestamp, returning its transition."""
    if state.get("removed") or at <= state.get("last", 0):
        return ""
    entry = state.get("entry")
    if not entry and not 40 < remaining <= 1440:
        return ""
    state["last"] = at
    state["quote"] = quote
    pending = [t for t in state.get("pending", []) if at-t <= 1800]
    if entry:
        points = favorite._american_implied_probability
        adverse = points(entry["price"]) - points(quote["price"])
        if remaining <= 40 and adverse >= 1-1e-8:
            issue, hard = "Closing price moved adversely by ≥1 raw implied point", True
        if issue and (hard or remaining <= 30 or (remaining > 40 and any(900 <= at-t <= 1800 for t in pending))):
            state["removed"] = issue
            return "removed"
        if issue:
            state["pending"] = pending + [at]
        else:
            state["pending"] = []
            if remaining <= 30:
                state["reviewed"] = True
        return ""
    if issue:
        state["pending"] = []
        return ""
    first = "first" not in state
    if first:
        state["first"] = quote
    if any(900 <= at-t <= 1800 for t in pending):
        state["entry"] = quote
        state["pending"] = []
        return "confirmed"
    state["pending"] = pending + [at]
    return "first" if first else ""


def handoff_issue(state, cutoff):
    if not state.get("reviewed"):
        return "No observed own-rule T−30 requalification"
    if cutoff-state["last"] > 720:
        return "Last eligible quote older than 12 minutes at handoff proxy"
    return ""


def first_confirmed_by_pathway(states):
    ordered = sorted((state["entry"]["at"], state["entry"]["side"], key, state["pathway"])
                     for key, state in states.items() if state.get("entry"))
    winners = {}
    for _, _, key, pathway in ordered:
        winners.setdefault(pathway, key)
    return winners


def first_confirmed_lane(states):
    entries = [(key, states[key]["entry"]) for key in first_confirmed_by_pathway(states).values()]
    if not entries:
        return None
    earliest = min(row["at"] for _, row in entries)
    tied = [(key, row) for key, row in entries if row["at"] == earliest]
    return min(key for key, _ in tied) if len({row["side"] for _, row in tied}) == 1 else ""
