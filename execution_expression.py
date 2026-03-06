"""
Red Fox v3.2 — Execution Expression Layer

Post-certification instrument routing.
Answers: "What is the best instrument to express this edge?"

The No-Escalation Rule (permanent, hard):
  The execution layer NEVER upgrades aggressiveness.
  It may only preserve the signal or route to more conservative instrument.
  Spread signal → ML is NEVER allowed.
"""

import engine_config_v3 as C


def compute_expression(row: dict, decision: str, score: float) -> dict:
    """Route to optimal instrument: ML / SPREAD / PUCK_LINE / RUN_LINE / PASS.

    Args:
        row: Full row dict
        decision: Certification result (STRONG_BET / BET / LEAN / NO_BET)
        score: v3.2 final_score

    Returns:
        dict with:
            expression: str (ML / SPREAD / PUCK_LINE / RUN_LINE / PASS)
            expression_reason: str
    """
    sport = (row.get("sport") or "").lower()
    market = (row.get("market_display") or row.get("market") or "").upper()
    l1_present = _bool(row.get("l1_available", row.get("l1_present", False)))
    path = (row.get("l1_path_behavior") or "UNKNOWN").upper()
    current_odds = _num(row.get("current_odds", 0))

    # NO_BET → no expression needed
    if decision == "NO_BET":
        return _result("PASS", "NO_BET decision")

    # ── NO-ESCALATION RULE ──
    # Spread/Total signals NEVER route to ML
    if market in ("SPREAD", "TOTAL"):
        # Spread stays spread, or routes to PASS
        return _handle_spread_signal(row, decision, score, sport, market, current_odds)

    # ── ML SIGNAL ROUTING ──
    if market in ("MONEYLINE", "ML"):
        return _handle_ml_signal(row, decision, score, sport, current_odds,
                                l1_present, path)

    # Unknown market → preserve
    return _result(market, "Unknown market — preserved")


def _handle_spread_signal(row, decision, score, sport, market, current_odds):
    """Spread/Total signals: stay or PASS. Never escalate to ML."""

    # LEAN + expensive odds → PASS
    if decision == "LEAN":
        if current_odds != 0 and current_odds < -200:
            return _result("PASS", f"LEAN + expensive odds ({current_odds})")

    # Spread stays spread
    return _result(market, "Spread signal preserved")


def _handle_ml_signal(row, decision, score, sport, current_odds,
                      l1_present, path):
    """ML signal: route based on price discipline."""

    # Determine if dog or favorite
    is_dog = current_odds > 0
    is_fav = current_odds < 0 and current_odds != 0

    # No odds data → preserve ML
    if current_odds == 0:
        return _result("ML", "No odds data — ML preserved")

    # ── UNDERDOG ROUTING ──
    if is_dog:
        return _route_dog(row, decision, score, sport, current_odds,
                         l1_present, path)

    # ── FAVORITE ROUTING ──
    if is_fav:
        return _route_favorite(row, decision, score, sport, current_odds)

    return _result("ML", "Even odds — ML preserved")


def _route_dog(row, decision, score, sport, odds, l1_present, path):
    """Underdog price discipline with tier compression."""
    abs_odds = abs(odds)
    routing = C.SPORT_ROUTING.get(sport, C.SPORT_ROUTING.get("nba"))
    alt = routing["alt"]

    # ── PASS conditions (spec section 10.5) — checked BEFORE tier routing ──
    # LEAN + long dog > +200 → PASS
    if decision == "LEAN" and abs_odds > 200:
        return _result("PASS", f"LEAN + long dog +{abs_odds}")

    # L1 absent + dog > +220 → PASS
    if not l1_present and abs_odds > 220:
        return _result("PASS", f"L1 absent + dog +{abs_odds}")

    # UFC: ML only, no alternate instrument
    if sport == "ufc":
        return _result("ML", f"UFC ML dog +{abs_odds}")

    # Check dog tier thresholds
    for tier_min, tier_max, min_score in C.DOG_TIERS:
        if tier_min <= abs_odds <= tier_max:
            if min_score is None:
                # Extreme dog (+301+): STRONG + L1 + HELD/EXTENDED required
                if (decision == "STRONG_BET" and l1_present
                        and path in ("HELD", "EXTENDED")):
                    return _result("ML", f"Extreme dog +{abs_odds} — all gates pass")
                else:
                    return _result("PASS",
                                   f"Extreme dog +{abs_odds} — requires STRONG+L1+HELD/EXTENDED")
            elif score >= min_score:
                return _result("ML", f"Dog +{abs_odds} — score {score} meets {min_score}")
            else:
                # Route to sport-specific alternate
                if alt:
                    return _result(alt,
                                   f"Dog +{abs_odds} — score {score} < {min_score}, route to {alt}")
                else:
                    return _result("PASS",
                                   f"Dog +{abs_odds} — score {score} < {min_score}, no alternate")
            break

    return _result("ML", f"Dog +{abs_odds} — acceptable")


def _route_favorite(row, decision, score, sport, odds):
    """Favorite price discipline."""
    abs_odds = abs(odds)
    routing = C.SPORT_ROUTING.get(sport, C.SPORT_ROUTING.get("nba"))
    alt = routing["alt"]

    # Comfortable range: -110 to -180
    if abs_odds <= abs(C.FAV_COMFORTABLE_MAX):
        return _result("ML", f"Favorite {odds} — comfortable range")

    # Elevated: -181 to -240 — compress one tier
    if abs_odds <= abs(C.FAV_COMPRESS_MAX):
        # STRONG → BET compression (don't output STRONG for elevated favorites)
        compressed = decision == "STRONG_BET"
        return _result("ML",
                       f"Favorite {odds} — elevated" +
                       (" (STRONG compressed to BET)" if compressed else ""))

    # Heavy favorite: -241+ — prefer SPREAD
    if alt:
        return _result(alt, f"Heavy favorite {odds} — route to {alt}")
    else:
        return _result("PASS", f"Heavy favorite {odds} — no alternate instrument")


def _handle_lean_pass(decision, current_odds):
    """LEAN + expensive price → PASS."""
    if decision != "LEAN":
        return None
    if current_odds > 0 and abs(current_odds) > 200:
        return _result("PASS", f"LEAN + long dog +{abs(current_odds)}")
    if current_odds < 0 and abs(current_odds) > 200:
        return _result("PASS", f"LEAN + expensive favorite {current_odds}")
    return None


def _result(expression, reason):
    return {
        "expression": expression,
        "expression_reason": reason,
    }


def _num(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    try:
        return bool(val)
    except (ValueError, TypeError):
        return False
