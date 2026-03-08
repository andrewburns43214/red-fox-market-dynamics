"""
Red Fox v3.2 — Engine Configuration
All magic numbers from the scoring spec in one place.
Single source of truth. Any change requires a version bump.
"""

V3_VERSION = "v3.3b"

# ─── FIXED BASE ───
BASE = 50  # Fixed. All sports. No exceptions.

# ─── COMPONENT 1: SHARP SIGNAL [-15, +20] ───
SHARP_MIN = -15
SHARP_MAX = 20

# Magnitude scaling by market
SHARP_MAG_SPREAD_MULT = 4.0
SHARP_MAG_SPREAD_CAP = 16.0
SHARP_MAG_TOTAL_MULT = 3.0
SHARP_MAG_TOTAL_CAP = 12.0
SHARP_MAG_ML_MULT = 0.20
SHARP_MAG_ML_CAP = 14.0

# Timing bucket dampening on magnitude
SHARP_TIMING_MULT = {"EARLY": 1.00, "MID": 0.90, "LATE": 0.60}

# Agreement multiplier — graduated (Pinnacle-weighted, not gated)
SHARP_AGREEMENT_BOOST = 1.15     # Pinnacle + cluster agree = strongest
SHARP_AGREEMENT_MODERATE = 1.10  # Pinnacle alone OR cluster without Pinnacle
SHARP_AGREEMENT_SLIGHT = 1.05   # Cross-tier (1 sharp + 1 supporting)
SHARP_AGREEMENT_DAMPEN = 0.90   # Weak/noise

# Path behavior adjustments (v3.2 swapped REVERSED/OSCILLATED)
SHARP_PATH_ADJ = {
    "HELD": 2.0,
    "EXTENDED": 2.0,
    "BUYBACK": 0.5,
    "UNKNOWN": 0.0,
    "OSCILLATED": -2.0,
    "REVERSED": -3.0,
}

# Key number bonus (NFL/NCAAF only)
SHARP_KEY_NUMBER_BONUS = 2.0
KEY_NUMBER_SPORTS = {"nfl", "ncaaf"}

# Direction flip (sharps oppose side)
SHARP_DIRECTION_FLIP_MULT = -0.75

# ─── COMPONENT 2: CONSENSUS VALIDATION [-10, +18] ───
CONSENSUS_MIN = -10
CONSENSUS_MAX = 18

# Agreement tiers (v3.2 raised)
CONSENSUS_TIERS = [
    (0.75, 14),   # strong
    (0.55, 7),    # moderate
    (0.35, 1),    # weak
    (0.00, -8),   # rejects
]

# Dispersion multipliers
CONSENSUS_DISPERSION_MULT = {
    "TIGHT": 1.25,
    "NORMAL": 1.00,
    "WIDE": 0.70,
    "VERY_WIDE": 0.40,
}

# Dispersion thresholds (for classifying TIGHT/NORMAL/WIDE/VERY_WIDE)
CONSENSUS_DISPERSION_THRESHOLDS_SPREAD = {
    "TIGHT": 0.3, "NORMAL": 1.0, "WIDE": 2.0,
}
CONSENSUS_DISPERSION_THRESHOLDS_TOTAL = {
    "TIGHT": 0.5, "NORMAL": 1.5, "WIDE": 3.0,
}

# Trend adjustment
CONSENSUS_TREND_ADJ = {"TIGHTENING": 2.0, "STABLE": 0.0, "WIDENING": -2.0}

# Stale price bonus
CONSENSUS_STALE_LARGE = 2.5   # DK ≥ 1.0 pt behind
CONSENSUS_STALE_SMALL = 1.0   # DK 0.5–0.9 pt behind
CONSENSUS_STALE_LARGE_THRESHOLD = 1.0
CONSENSUS_STALE_SMALL_THRESHOLD = 0.5

# Book count guard
CONSENSUS_BOOK_GUARD = [
    (10, 1.00),  # ≥ 10 books
    (5, 0.80),   # 5-9 books
    (0, 0.60),   # < 5 books
]

# ─── COMPONENT 3: RETAIL CONTEXT [-8, +8] ───
RETAIL_MIN = -8
RETAIL_MAX = 8

# Divergence signal scaling by market
RETAIL_DIV_SPREAD_MULT = 0.30
RETAIL_DIV_SPREAD_CAP = 6.0
RETAIL_DIV_TOTAL_MULT = 0.22
RETAIL_DIV_TOTAL_CAP = 4.0
RETAIL_DIV_ML_MULT = 0.25
RETAIL_DIV_ML_CAP = 5.0

# Negative divergence dampening
RETAIL_NEG_DIV_MULT = -0.40

# Crowding penalty (v3.2 raised to 72%)
RETAIL_CROWDING_BETS_THRESHOLD = 72
RETAIL_CROWDING_MONEY_THRESHOLD = 72
RETAIL_CROWDING_PENALTY = -5.0

# Parlay distortion (ML only)
RETAIL_PARLAY_PENALTY = -4.0
RETAIL_PARLAY_MONEY_THRESHOLD = 80  # money_pct >80%
RETAIL_PARLAY_ODDS_THRESHOLD = -150  # favorite < -150

# DK line confirmation
RETAIL_DK_CONFIRM_BONUS = 1.5
RETAIL_DK_OPPOSE_PENALTY = -1.5
RETAIL_DK_CONFIRM_MIN_MOVE = 0.5  # ≥ 0.5 pt DK move required

# Sample credibility
RETAIL_SAMPLE_CREDIBILITY = [
    (20, 1.00),  # ≥ 20%
    (10, 0.80),  # 10-19%
    (0, 0.65),   # < 10%
]

# ML instrument multiplier (permanent, all sports)
RETAIL_ML_MULTIPLIER = 0.60

# ─── COMPONENT 4: TIMING MODIFIER [-5, +1] ───
TIMING_MIN = -5
TIMING_MAX = 1

TIMING_EARLY = -2
TIMING_MID_BASE = 0
TIMING_MID_BOOST = 1  # Only when L1 present AND path HELD or EXTENDED
TIMING_LATE_BASE = -3
TIMING_LATE_REVERSED = -5  # LATE + REVERSED or OSCILLATED

# NBA exception
TIMING_NBA_LATE_CAP = -3  # NBA LATE never worse than -3

# Sport timing windows (minutes before game start)
TIMING_WINDOWS = {
    "nba":   {"EARLY": 480, "LATE": 60},
    "nhl":   {"EARLY": 480, "LATE": 60},
    "mlb":   {"EARLY": 480, "LATE": 60},
    "nfl":   {"EARLY": 1440, "LATE": 360},
    "ncaab": {"EARLY": 480, "LATE": 60},
    "ncaaf": {"EARLY": 1440, "LATE": 360},
    "ufc":   {"EARLY": 480, "LATE": 60},
}

# ─── COMPONENT 5: CROSS-MARKET SANITY [-4, +4] ───
CROSS_MARKET_AGREE = 4
CROSS_MARKET_CONTRADICT = -4
CROSS_MARKET_EXEMPT_SPORTS = {"ufc"}
CROSS_MARKET_MLB_RUNLINE_EXEMPT = True  # MLB run line excluded

# ─── CERTIFICATION THRESHOLDS ───
STRONG_SCORE_MIN = 70
BET_SCORE_MIN = 67
LEAN_SCORE_MIN = 60

STRONG_EDGE_MIN_SIDES = 10   # spread / ML
STRONG_EDGE_MIN_TOTAL = 12   # total
BET_EDGE_MIN_SIDES = 10
BET_EDGE_MIN_TOTAL = 12

# STRONG gates
STRONG_STREAK_MIN = 2
NCAAB_STREAK_MIN = 3
STRONG_STABILITY_DELTA = 3.0
NCAAB_STABILITY_DELTA = 2.0
STRONG_BLOCKED_PATHS = {"REVERSED", "OSCILLATED"}
STRONG_EARLY_BLOCK_SPORTS = {"ncaab", "ncaaf"}

# L1-absent caps
L1_ABSENT_STRONG_DISABLED = True
L1_ABSENT_L2_WEAK_CAP = 66
L1_ABSENT_L2_WEAK_THRESHOLD = 0.55  # l2_agreement below this = weak

# ─── EXECUTION EXPRESSION ───

# Dog price tiers: (min_odds, max_odds, min_score_for_ML)
DOG_TIERS = [
    (100, 160, 67),
    (161, 220, 69),
    (221, 300, 72),
    (301, 9999, None),  # None = requires STRONG + L1 + HELD/EXTENDED
]

# Favorite price tiers
FAV_COMFORTABLE_MAX = -180      # -110 to -180: ML acceptable
FAV_COMPRESS_MAX = -240         # -181 to -240: compress one tier
# -241 and beyond: prefer SPREAD

# Sport instrument routing: (sport, preferred_alternate, dog_threshold, fav_threshold)
SPORT_ROUTING = {
    "nba":   {"alt": "SPREAD",    "dog_ml_max": 220, "fav_ml_min": -241},
    "nhl":   {"alt": "PUCK_LINE", "dog_ml_max": 200, "fav_ml_min": -220},
    "mlb":   {"alt": "RUN_LINE",  "dog_ml_max": 180, "fav_ml_min": -200},
    "nfl":   {"alt": "SPREAD",    "dog_ml_max": 220, "fav_ml_min": -241},
    "ncaab": {"alt": "SPREAD",    "dog_ml_max": 220, "fav_ml_min": -999},
    "ncaaf": {"alt": "SPREAD",    "dog_ml_max": 220, "fav_ml_min": -999},
    "ufc":   {"alt": None,        "dog_ml_max": 9999, "fav_ml_min": -999},
}

# ML price band credibility multipliers
ML_PRICE_CREDIBILITY = [
    (33, 1.00),   # ≥ 33% implied prob (favorites / short dogs)
    (25, 0.85),   # 25-32%
    (20, 0.70),   # 20-24%
    (15, 0.60),   # 15-19%
    (0, 0.50),    # < 15% implied prob (extreme dogs)
]

# ─── PATTERNS (output labels only, never affect score) ───
PATTERNS = {
    "SHARP_REVERSAL": "L1 moved + public heavy opposite + path HELD/EXTENDED",
    "STALE_PRICE": "DK lags consensus ≥1pt + L1 confirmed direction",
    "FREEZE_PRESSURE": "L1 moved + L2 strongly aligned + no DK response",
    "PUBLIC_DRIFT": "Heavy public bets + money + line toward public side",
    "CONSENSUS_HOLD": "L2 strongly aligned without clear L1 move",
    "RETAIL_CROWD": "Extreme public concentration, no sharp support",
    "NEUTRAL": "No strong pattern in any bucket",
}

# ─── CROSS-SECTIONAL CONSENSUS (single-snapshot fallback) ───
# 4-tier scoring: reserve +10 for time-series only, cross-section caps at +8
CROSS_PINN_TIERS = [
    # (min_gap, min_books, score)  — evaluated top-down, first match wins
    (3.0, 10, 8.0),    # very strong: ≥3 pts, 10+ books
    (2.0, 8,  6.0),    # strong: ≥2 pts, 8+ books
    (1.5, 8,  4.0),    # moderate: ≥1.5 pts, 8+ books
    (1.0, 5,  2.0),    # weak: ≥1 pt, 5+ books
]
CROSS_TIGHT_DAMPENING = 0.5       # Reduce when TIGHT + small gap
CROSS_VERY_WIDE_DAMPENING = 0.4   # Noisy market

# ─── RETAIL DAMPENING WHEN L1/L2 ABSENT ───
RETAIL_L1_ABSENT_MULT = 0.7       # Soft dampen retail when unsupported

# NHL puck line dampening on Sharp component
NHL_PUCK_LINE_SHARP_MULT = 0.85

# NHL retail sample credibility multiplier
NHL_RETAIL_SAMPLE_MULT = 0.80

# NCAAB adjustments
NCAAB_PUBLIC_HEAVY_THRESHOLD = 75      # vs 72 default
NCAAB_RETAIL_DIV_MIN = 20             # minimum D for meaningful signal
NCAAB_CONSENSUS_EFFECTIVE_BUMP = 0.05  # interpretation note, not formula change

# ─── COMPONENT 6: MARKET REACTION [-4, +12] ───
MARKET_REACTION_MIN = -4
MARKET_REACTION_MAX = 12

# Book-initiated movement (no public pressure)
MR_BOOK_MOVE_THRESHOLD = 0.5        # min effective_move_mag to qualify
MR_BOOK_BETS_CEILING = 60           # bets_pct must be below
MR_BOOK_MONEY_CEILING = 65          # money_pct must be below
MR_BOOK_INITIATED_BONUS = 4.0
