"""
Red Fox v3.2 — Path Behavior Classifier

Classifies L1 line trajectory from row_state tracking data.
Tuned to actual DK scraper cadence (every 10 min = 6 ticks/hour).

Uses existing fields from row_state.csv:
  - line_settled_ticks: consecutive ticks with no movement
  - line_dir_changes:   count of direction reversals since open
  - line_last_dir:      last movement direction (-1, 0, +1)
  - line_max_move:      maximum movement from open (absolute)

Also uses:
  - effective_move_mag or line_move_open: current net movement from open
  - ticks_since_open:   total ticks tracked (for UNKNOWN guard)
"""


def classify_path_behavior(row: dict) -> str:
    """Classify line trajectory into v3.2 path behavior labels.

    Returns one of:
        HELD        Move committed and held (3+ hrs stable, no reversal)
        EXTENDED    Still moving same direction (active, no reversal)
        BUYBACK     Partial reversal (moved back but net still >50% of max)
        REVERSED    Faded back (net move <50% of max)
        OSCILLATED  Bouncing (2+ direction changes)
        UNKNOWN     Insufficient data
    """
    settled = _num(row.get("line_settled_ticks", 0))
    dir_changes = _num(row.get("line_dir_changes", 0))
    last_dir = _num(row.get("line_last_dir", 0))
    max_move = abs(_num(row.get("line_max_move", 0)))
    ticks = _num(row.get("ticks_since_open", 0))

    # Current net movement from open
    current_move = abs(_num(row.get("effective_move_mag",
                           row.get("line_move_open", 0))))

    # ── UNKNOWN: insufficient data ──
    # Need at least 3 ticks (~30 min) of tracking and some movement
    if ticks < 3 or (max_move == 0 and settled == 0 and dir_changes == 0):
        return "UNKNOWN"

    # ── OSCILLATED: 2+ direction changes ──
    if dir_changes >= 2:
        return "OSCILLATED"

    # ── REVERSED: faded back (net move < 50% of max) ──
    if dir_changes >= 1:
        if max_move > 0 and current_move < max_move * 0.5:
            return "REVERSED"
        else:
            return "BUYBACK"

    # No direction changes from here (dir_changes == 0)

    # ── HELD: stable for 3+ hours after moving ──
    # 18 ticks × 10 min = 3 hours of stability
    if settled >= 18 and max_move > 0:
        return "HELD"

    # ── EXTENDED: still actively moving same direction ──
    if last_dir != 0 and settled < 18:
        return "EXTENDED"

    # ── Fallback: moved but in limbo ──
    # Has some movement but not enough ticks to classify as HELD
    if max_move > 0 and settled > 0:
        # Some stability but < 3 hours — lean toward HELD if > 1 hour
        if settled >= 6:
            return "HELD"
        return "EXTENDED"

    return "UNKNOWN"


def _num(val) -> float:
    """Safely convert to float, defaulting to 0."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
