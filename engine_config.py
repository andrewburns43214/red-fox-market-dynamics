"""
Centralized configuration for the Red Fox engine.
All magic numbers, file paths, API keys, and layer weights in one place.

CHANGELOG
---------
v2.1 (2026-03-05) — Layer Trust + Sharp Certified + STRONG_BET Rework
  - Removed hard layer caps (L123=100, L13=85, L23=80, L3_ONLY=75).
    Layer mode is now UI badge only; all rows capped at 100 universally.
  - Sharp Certified signal (NONE / HALF / FULL):
      HALF: Pinnacle moved meaningfully (mag>=0.3) in clear direction.
      FULL: Pinnacle + 1 other sharp book agree. Bonus +8-12, replaces l1_adj.
      DK response amplifier (1.3x) when DK moved same direction.
  - STRONG_BET 3 paths:
      Path 1 (Pattern): A/D/G + score>=70 + edge>=10 + persist>=2
      Path 2 (Sharp Certified FULL): score>=70 + edge>=10 + persist>=2
      Path 3 (Score-only): score>=75 + edge>=12 + persist>=3
  - Dashboard: green pulsing "SHARP check" badge (FULL), yellow "SHARP ~" (HALF)

v2.0 (2026-03-05) — Scoring Signal Overhaul
  - 3-layer scoring: L1 (Pinnacle sharp) + L2 (31-book consensus) + L3 (DK retail)
  - Book response philosophy: read the BOOK's line movement, not DK bettors' money
  - Continuous signals, line trajectory, ML vs Spread cross-check
  - Pattern detection A-G with floors/caps/STRONG gates

v2.0 Audit (2026-03-05) — 6 files, 14 fixes
  Critical (P1):
    - Outcome resolution (main.py): deterministic team1=away, team2=home ordering
    - Key number bonus (scoring_v2.py): restricted to NFL/NCAAF only
    - MLB team aliases (team_aliases.py): added 30 abbreviation entries
  High (P2):
    - Reversed team match (canonical_match.py): non-UFC scores 0.0 for reversed order
    - Config extraction: moved score floors, public heavy, cross-check, decay, B2B to config
    - Budget guard (main.py): actually stops pulls when budget exhausted
    - Timezone validation (main.py): _validate_iso_tz() appends Z to naive datetimes
  Medium (P3):
    - BOTH_B2B (scoring_v2.py): explicitly handled with 0 adjustment
    - UFC market filter (merge_layers.py): strips SPREAD/TOTAL rows for UFC
    - Pattern B/C (engine_config.py): B gets explicit bonus:0, C renamed RETAIL_ONLY
    - Bare except (main.py): replaced with except (ValueError, TypeError)
  Verified non-issues: stale detection on fuzzy L2, NCAAB string formatting
"""
import os

# ─── VERSION ───
LOGIC_VERSION = "v2.1"

# ─── FILE PATHS ───
DATA_DIR = "data"
SNAPSHOT_CSV = os.path.join(DATA_DIR, "snapshots.csv")
OPEN_REGISTRY_CSV = os.path.join(DATA_DIR, "open_registry.csv")
ROW_STATE_CSV = os.path.join(DATA_DIR, "row_state.csv")
SIGNAL_LEDGER_CSV = os.path.join(DATA_DIR, "signal_ledger.csv")
DASHBOARD_CSV = os.path.join(DATA_DIR, "dashboard.csv")
DASHBOARD_HTML = os.path.join(DATA_DIR, "dashboard.html")

# Layer 1/2 data files
L1_SHARP_CSV = os.path.join(DATA_DIR, "l1_sharp.csv")
L1_OPEN_REGISTRY_CSV = os.path.join(DATA_DIR, "l1_open_registry.csv")
L2_CONSENSUS_CSV = os.path.join(DATA_DIR, "l2_consensus.csv")
L2_CONSENSUS_AGG_CSV = os.path.join(DATA_DIR, "l2_consensus_agg.csv")
L1_CACHE_JSON = os.path.join(DATA_DIR, "l1_cache.json")
L2_CACHE_JSON = os.path.join(DATA_DIR, "l2_cache.json")
MATCH_FAILURES_CSV = os.path.join(DATA_DIR, "match_failures.csv")
SCORE_COMPARISON_CSV = os.path.join(DATA_DIR, "score_comparison.csv")

# ─── API CONFIGURATION ───
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Maps our sport keys to The-Odds-API sport keys
API_SPORT_MAP = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "ncaaf": "americanfootball_ncaaf",
    "ncaab": "basketball_ncaab",
    "nhl": "icehockey_nhl",
    "mlb": "baseball_mlb",
    "ufc": "mma_mixed_martial_arts",
}

# Reverse map
API_SPORT_MAP_REVERSE = {v: k for k, v in API_SPORT_MAP.items()}

# Sharp books for Layer 1 (order = priority)
L1_SHARP_BOOKS = ["pinnacle"]

# Regions to fetch — us for domestic books, eu for Pinnacle/offshore sharp books
L2_CONSENSUS_REGIONS = ["us", "eu"]

# Cache TTL (seconds) - use cached data if API fails and cache is this fresh
CACHE_TTL_SECONDS = 1800  # 30 minutes

# ─── SCORING WEIGHTS ───

# Layer contribution ranges (unified engine)
L1_MAX_ADJUSTMENT = 10.0       # was L1_MAX_CONTRIBUTION = 18.0
L1_MIN_ADJUSTMENT = -5.0       # NEW: bidirectional (penalizes when sharp opposes DK)
L2_MAX_POSITIVE_ADJ = 7.0      # was L2_MAX_POSITIVE = 10.0
L2_MAX_NEGATIVE_ADJ = -5.0     # was L2_MAX_NEGATIVE = -8.0
LINE_DIFF_MAX_BONUS = 8.0      # NEW: DK vs consensus/Pinnacle differential

# Legacy aliases (for backward compatibility during transition)
L1_MAX_CONTRIBUTION = L1_MAX_ADJUSTMENT
L2_MAX_POSITIVE = L2_MAX_POSITIVE_ADJ
L2_MAX_NEGATIVE = L2_MAX_NEGATIVE_ADJ
L3_MAX_POSITIVE = 10.0
L3_MAX_NEGATIVE = -10.0

# Layer mode score caps — REMOVED (v2.1)
# Layers now contribute to score but don't cap it. Layer mode is UI-only.
# Old values preserved for reference:
# SCORE_CAP_L123 = 100, SCORE_CAP_L13 = 85, SCORE_CAP_L23 = 80, SCORE_CAP_L3_ONLY = 75
SCORE_CAP_UNIVERSAL = 100

# Pattern bonuses/penalties
PATTERN_EFFECTS = {
    "A": {"bonus": 5, "strong_eligible": True, "label": "SHARP_VS_PUBLIC"},
    "B": {"bonus": 0, "cap": 70, "strong_eligible": False, "label": "RETAIL_ALIGNMENT"},
    "C": {"bonus": -5, "strong_eligible": False, "label": "RETAIL_ONLY"},
    "D": {"bonus": 4, "strong_eligible": True, "label": "STALE_PRICE"},
    "E": {"bonus": -6, "cap": 65, "strong_eligible": False, "label": "CONSENSUS_REJECTS"},
    "F": {"bonus": -8, "strong_eligible": False, "label": "LATE_SNAP_WARNING"},
    "G": {"bonus": 4, "strong_eligible": True, "label": "REVERSE_LINE_MOVE"},
}

# ─── RLM THRESHOLDS ───
RLM_BETS_THRESHOLD = 60     # bets% must be at least this on one side
RLM_MONEY_GAP_MIN = 15      # minimum bets% - money% gap for RLM
RLM_L2_AGREEMENT_MIN = 0.5  # L2 must agree at this level
RLM_MOVE_EXHAUSTION = 3.0   # if DK already moved >= this toward public side, dampen RLM

# ─── DK RULES THRESHOLDS (v2.0) ───

# ML instrument credibility (was 0.80 in v1.2, now 0.60)
DK_ML_INST_MULT = 0.60

# DK divergence thresholds (higher than v1.2 because DK is retail)
DK_DIVERGENCE_THRESHOLD = 15       # was 8 in v1.2
DK_ML_DIVERGENCE_THRESHOLD = 20    # ML needs even stronger divergence

# Retail alignment penalty threshold
DK_RETAIL_ALIGN_BETS_MIN = 70
DK_RETAIL_ALIGN_MONEY_MIN = 70
DK_RETAIL_ALIGN_PENALTY = -5

# ML-only penalty (ML moved, spread didn't)
DK_ML_ONLY_PENALTY = -3

# ─── MOVE SPEED THRESHOLDS (Layer 1) ───
FAST_SNAP_SPEED = 2.0       # pts/hour for spreads
SLOW_GRIND_SPEED = 0.5      # pts/hour for spreads
FAST_SNAP_EARLY_BONUS = 3
FAST_SNAP_LATE_PENALTY = -4
SLOW_GRIND_PENALTY = -2

# ─── DISPERSION THRESHOLDS (Layer 2) ───
DISPERSION_TIGHT_SPREAD = 0.3
DISPERSION_TIGHT_TOTAL = 0.5
DISPERSION_WIDE_SPREAD = 1.0
DISPERSION_WIDE_TOTAL = 1.5
DISPERSION_VERY_WIDE_SPREAD = 2.0
DISPERSION_VERY_WIDE_TOTAL = 3.0

DISPERSION_TIGHT_MULT = 1.20
DISPERSION_NORMAL_MULT = 1.00
DISPERSION_WIDE_MULT = 0.70
DISPERSION_VERY_WIDE_MULT = 0.40

# ─── DECISION THRESHOLDS ───
STRONG_BET_SCORE = 70
BET_SCORE = 67
LEAN_SCORE = 60

# Hysteresis: exit thresholds (must fall further to drop a tier)
STRONG_BET_EXIT = 66   # STRONG_BET stays until score < 66
BET_EXIT = 62           # BET stays until score < 62
LEAN_EXIT = 56          # LEAN stays until score < 56
NET_EDGE_MIN_SIDES = 10
NET_EDGE_MIN_TOTAL = 12

# ─── KEY NUMBERS ───
KEY_NUMBERS = {3, 7, 10, 14, 17}

# Score floors by pattern (minimum score based on signal quality)
SCORE_FLOORS = {
    "A": 50.0, "D": 50.0, "G": 50.0,
    "B": 45.0, "C": 40.0, "E": 40.0,
    "F": 40.0, "N": 40.0,
}

# Public heavy threshold (bets% above or below this = "public heavy")
PUBLIC_HEAVY_THRESHOLD = 65

# Cross-market check values
CROSS_CHECK_CONSISTENT = 1.0
CROSS_CHECK_CONTRADICTION = -2.0

# Momentum decay
DECAY_FLAT_TICK_START = 5
DECAY_MAX = -2.0

# B2B adjustment
B2B_SINGLE_ADJ = -1.0

# Line diff feature flag
LINE_DIFF_ENABLED = False

# ─── SHARP CERTIFIED ───
# Tier thresholds
SHARP_CERT_MIN_MAGNITUDE = 0.30       # normalized move size (0-1)
SHARP_CERT_LEADER_BOOKS = {"pinnacle", "betcris", "bookmaker.eu", "bookmaker"}  # sharpest books
SHARP_CERT_FULL_MIN_AGREEMENT = 2     # Pinnacle + at least 1 other sharp book
# Bonus ranges (replaces l1_adjustment when active)
SHARP_CERT_HALF_BONUS_MIN = 4.0
SHARP_CERT_HALF_BONUS_MAX = 6.0
SHARP_CERT_FULL_BONUS_MIN = 8.0
SHARP_CERT_FULL_BONUS_MAX = 12.0
# DK book response amplifier
SHARP_CERT_DK_RESPONSE_AMP = 1.3     # multiply bonus when DK moved same direction
SHARP_CERT_BONUS_HARD_CAP = 15.0     # absolute max after amplification

# ─── STRONG_BET PATHS ───
# Path 1 (Pattern): existing A/D/G + score ≥70 + edge ≥10 + persist ≥2
# Path 2 (Sharp Certified): SHARP FULL + score ≥70 + edge ≥10 + persist ≥2
# Path 3 (Score-only): score ≥75 + edge ≥12 + persist ≥3
STRONG_SCORE_ONLY_MIN = 75
STRONG_SCORE_ONLY_EDGE = 12
STRONG_SCORE_ONLY_PERSIST = 3

# ─── STALE ROW CLEANUP ───
STALE_TICK_THRESHOLD = 3  # expire rows not seen in this many ticks

# ─── SPORT SEASON CALENDAR ───
# (start_month, end_month) — wraps around year boundary (e.g. Oct-Jun = 10,6)
SPORT_SEASONS = {
    "ufc":   (1, 12),   # year-round
    "nba":   (10, 6),
    "nhl":   (10, 6),
    "mlb":   (3, 11),
    "ncaab": (11, 4),
    "ncaaf": (8, 1),
    "nfl":   (9, 2),
}


# ─── ODDSPAPI CONFIGURATION (Layer 1 primary) ───
ODDSPAPI_KEY = os.environ.get("ODDSPAPI_KEY", "")
ODDSPAPI_BASE_URL = "https://api.oddspapi.io/v4"
ODDSPAPI_CACHE_JSON = os.path.join(DATA_DIR, "oddspapi_cache.json")

# OddsPapi tournament IDs (discovered via /tournaments endpoint)
ODDSPAPI_TOURNAMENT_MAP = {
    "nba": 132,
    "ncaab": 648,
    "nhl": None,      # discover on first run
    "mlb": None,       # discover on first run
    "nfl": None,       # discover on first run
    "ufc": None,       # discover on first run
}

# Sharp books available on OddsPapi (6 books vs 1 on The-Odds-API)
ODDSPAPI_SHARP_BOOKS = [
    "pinnacle", "singbet", "sbobet",
    "betcris", "circasports", "bookmaker.eu",
]

# OddsPapi budget (separate from The-Odds-API)
ODDSPAPI_BUDGET_MONTHLY = 250
ODDSPAPI_BUDGET_RESERVE = 20

# OddsPapi monthly priority + pull schedule
# Each month: sports in PRIORITY ORDER (spend budget on top sports first).
# UFC is year-round and always included.
# Lower-priority sports past MAX_SPORTS get The-Odds-API Pinnacle fallback.
#
# Pull times (ET): 11:30, 15:30, 18:30 (NFL Sundays add 12:15)
# These match The-Odds-API L2 pulls so both layers are synchronized.
ODDSPAPI_PRIORITY_SCHEDULE = {
    #  Month  Priority Order (top 3 get OddsPapi, rest get The-Odds-API fallback)
    1:  ["nfl", "nba", "ncaab", "ufc", "nhl"],          # Jan: NFL playoffs, NBA, CBB, UFC, NHL
    2:  ["nfl", "nba", "ncaab", "ufc", "nhl"],          # Feb: Super Bowl, NBA, CBB, UFC, NHL
    3:  ["ncaab", "nba", "ufc", "nhl"],                  # Mar: March Madness, NBA, UFC, NHL
    4:  ["ncaab", "nba", "mlb", "ufc", "nhl"],           # Apr: CBB tourney, NBA, MLB opens, UFC
    5:  ["nba", "mlb", "ufc", "nhl"],                    # May: NBA playoffs, MLB, UFC, NHL playoffs
    6:  ["nba", "mlb", "ufc", "nhl"],                    # Jun: NBA Finals, MLB, UFC, NHL Finals
    7:  ["mlb", "ufc"],                                   # Jul: MLB, UFC
    8:  ["mlb", "ufc", "ncaaf"],                          # Aug: MLB, UFC, NCAAF starts
    9:  ["nfl", "ncaaf", "mlb", "ufc"],                   # Sep: NFL opens, NCAAF, MLB stretch, UFC
    10: ["nfl", "ncaaf", "mlb", "ufc", "nba", "nhl"],    # Oct: NFL, NCAAF, MLB postseason, UFC, NBA/NHL open
    11: ["nfl", "ncaaf", "nba", "ufc", "ncaab", "nhl"],  # Nov: NFL, NCAAF, NBA, UFC, CBB opens, NHL
    12: ["nfl", "ncaaf", "nba", "ufc", "ncaab", "nhl"],  # Dec: NFL, bowl games, NBA, UFC, CBB, NHL
}

# Max OddsPapi sports per pull (to stay within budget)
# 250 req/month ÷ 30 days ÷ 3 pulls/day = ~2.8 odds calls per pull
# Each sport = 1 odds call (fixture cached) → max 3 sports per pull
# Top 3 priority sports get OddsPapi; rest fall back to The-Odds-API
ODDSPAPI_MAX_SPORTS_PER_PULL = 3

# Pull schedule times (ET) — when odds_snapshot_all runs
# Weekdays:     11:30, 15:30, 18:30  (3 pulls)
# NFL Sundays:  11:00, 12:15, 15:30, 18:30  (4 pulls)
# Saturday:     11:30, 15:30, 18:30  (3 pulls, covers CFB)
PULL_SCHEDULE = {
    "weekday":    ["11:30", "15:30", "18:30"],
    "saturday":   ["11:30", "15:30", "18:30"],
    "nfl_sunday": ["11:00", "12:15", "15:30", "18:30"],
    "sunday":     ["11:30", "15:30", "18:30"],
}

# ─── ESPN CONFIGURATION ───
ESPN_CACHE_TTL_SECONDS = 1800  # 30 minutes
ESPN_SPORT_PATHS = {
    "nba": "basketball/nba",
    "nhl": "hockey/nhl",
    "mlb": "baseball/mlb",
    "nfl": "football/nfl",
    "ncaab": "basketball/mens-college-basketball",
    "ncaaf": "football/college-football",
}

# ─── API BUDGET (The-Odds-API for L2 consensus) ───
API_MONTHLY_BUDGET = 500          # free tier limit
API_BUDGET_RESERVE = 30           # stop pulling if fewer than this remain
API_PULLS_PER_DAY_DEFAULT = 3     # weekdays + non-NFL Sundays
API_PULLS_NFL_SUNDAY = 4          # NFL Sundays get an extra pull

# Pull schedule (ET hours) — used by cron documentation, not enforced in code
# Weekdays:     11:30, 15:30, 18:30
# NFL Sundays:  11:00, 12:15, 15:30, 18:30


def get_active_sports(month: int = None) -> list:
    """Return list of sport keys active for the given month (1-12)."""
    if month is None:
        from datetime import datetime, timezone
        month = datetime.now(timezone.utc).month

    active = []
    for sport, (start, end) in SPORT_SEASONS.items():
        if start <= end:
            # Normal range (e.g. Mar-Nov)
            if start <= month <= end:
                active.append(sport)
        else:
            # Wraps around year (e.g. Oct-Jun = 10,6)
            if month >= start or month <= end:
                active.append(sport)
    return active
