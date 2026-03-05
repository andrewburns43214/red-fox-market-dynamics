# Red Fox Market Intelligence — Engine Architecture v2.1
*Updated: 2026-03-05 — CLV tracking, weather integration, sport-specific context layers, injury scoring*

---

## EVOLUTION SUMMARY

| Version | Date | Architecture |
|---------|------|-------------|
| v1.0 | 2026-02-27 | Single-layer DK scoring. Color classification, additive signals, fixed base 50. |
| v1.1 | 2026-03-01 | STRONG_BET wiring, canonical keys, elig_map, persistence cap, SPREAD dampening. Still single-layer. |
| v1.2 | 2026-03-02 | Regime classifier, RLM signal, combined divergence multiplier, dynamic base, sport-specific rebalancing. Still single-layer. |
| v2.0 | 2026-03-04 | 3-Layer model. L1 (sharp books via OddsPapi), L2 (31-book consensus via The-Odds-API), L3 (DK retail behavior). Pattern detection system. Cross-layer interaction scoring. Layer mode caps. Full redesign. |
| **v2.1** | **2026-03-05** | **Situational intelligence layer. CLV tracking for engine validation. Weather scoring (Open-Meteo). ESPN injury dampening. Sport-specific context: MLB pitching + park factors, NHL goalie confirmation, NCAAB rankings. Score formula now has 11 adjustment components.** |

**What changed from v2.0 → v2.1:**
- v2.0 had 7 adjustment components in the score formula (L1, L2, pattern, cross-market, line diff, decay, B2B)
- v2.1 adds 4 more: injury_adj, weather_adj, sport_context_adj, and wires them through scoring
- CLV tracking enables engine performance validation beyond win/loss variance
- Weather data from Open-Meteo (free, no key) affects outdoor sports (NFL, NCAAF, MLB)
- MLB gets starting pitcher quality scoring + park factors for all 30 teams
- NHL gets goalie confirmation/backup detection from ESPN
- NCAAB gets AP/NET ranking differential scoring
- Dashboard badges for all new signals (pitcher ERA, goalie status, park factor, weather, rankings)
- **Scoring recalibration:** `effective_move_mag` combines line number movement + juice/odds movement into a single magnitude. 45% of DK line changes were juice-only (e.g., -115→-105) where the line NUMBER didn't change but odds shifted — the engine was blind to these. Juice converted to line-equivalent: `min(3.0, abs(odds_move) / 15.0)`.
- **Boosted confirmed signals:** Divergence cap 12→16 (sides), Color bonus 8→12 (DARK_GREEN+book), Line trajectory bonuses doubled, 2-tier line movement multiplier (small moves 2.5x, large moves 3.0x)
- **Layer mode caps removed:** All modes use universal cap of 100. Layers contribute via adjustments, not ceilings.
- **STRONG_BET 3 paths:** Pattern-driven, Sharp Certified, Score-only (higher bar). L3_ONLY explicitly guarded.
- **Silent failure logging:** All bare `except: pass` blocks now print warnings. Layer merge failure is loud.
- **Score decomposition:** Dashboard now includes `v2_dk_base`, `v2_pattern_bonus`, `v2_decay` for full auditability.
- **Tick shadowing fix:** `metrics_update.py` `dashboard_tick` now uses `global` keyword correctly.
- **5 invariant tests:** `test_invariants.py` — score 0-100, no STRONG without edge, L3_ONLY guard, decision monotonicity, merge failure logging.

---

## 1. THE THREE LAYERS

### Layer 1 — Sharp Books (OddsPapi)
**Source:** OddsPapi API → `l1_sharp.csv`
**Books:** Pinnacle, Bookmaker.eu (2 of 6 available — Pinnacle is primary sharp source)
**Pull schedule:** 3x daily (11:30, 15:30, 18:30 ET), NFL Sundays add 12:15 ET

**What L1 provides:**
- `l1_move_dir` — direction of sharp line movement (+1 toward side, -1 against, 0 none)
- `l1_move_magnitude` — normalized magnitude of move (0 to 1)
- `l1_sharp_agreement` — do both sharp books agree? (0 or 1)
- `l1_agreement_mult` — agreement multiplier (1.0 to 2.0)
- `l1_move_speed` — points per hour of movement
- `l1_speed_label` — FAST_SNAP / SLOW_GRIND / empty
- `l1_stability` — how stable the move is (0 to 1)
- `l1_key_number_cross` — did the move cross a key number? (3, 7, 10, 14, 17)
- `l1_sharp_strength` — composite strength (0 to 1)
- `l1_limit_confidence` — limit size as confidence proxy
- `l1_n_books` — number of sharp books with data

**Feature extraction:** `l1_features.py` → `compute_l1_features(sport)`
**Open registry:** `l1_open_registry.csv` tracks opening lines for move detection

---

### Layer 2 — Consensus Market (The-Odds-API)
**Source:** The-Odds-API → `l2_consensus.csv` → `l2_consensus_agg.csv`
**Books:** 31 books across US + EU regions (Pinnacle, DraftKings, FanDuel, BetMGM, Bovada, William Hill, Unibet, etc.)
**Budget:** 500 req/month free tier, 30-request reserve guard
**Pull schedule:** Synchronized with L1 (same 3x daily)

**What L2 provides:**
- `l2_n_books` — number of books in consensus (typically 15-31)
- `l2_consensus_line` — median line across all books
- `l2_consensus_odds` — median odds
- `l2_dispersion` — standard deviation of lines across books
- `l2_dispersion_label` — TIGHT / NORMAL / WIDE / VERY_WIDE
- `l2_dispersion_mult` — confidence multiplier (0.40 to 1.20)
- `l2_dispersion_trend` — TIGHTENING / STABLE / WIDENING
- `l2_pinn_vs_consensus` — Pinnacle line minus consensus (sharp divergence)
- `l2_pinn_line` — Pinnacle's specific line
- `l2_consensus_agreement` — % of books moving in L1's direction (0 to 1)
- `l2_stale_price_flag` — DK line lagging consensus by > threshold
- `l2_stale_price_gap` — how many points DK is behind
- `l2_validation_strength` — composite validation score (0 to 1)

**Dispersion thresholds:**

| Category | SPREAD | TOTAL | Multiplier |
|----------|--------|-------|------------|
| TIGHT | ≤ 0.3 | ≤ 0.5 | 1.20 |
| NORMAL | ≤ 1.0 | ≤ 1.5 | 1.00 |
| WIDE | ≤ 2.0 | ≤ 3.0 | 0.70 |
| VERY_WIDE | > 2.0 | > 3.0 | 0.40 |

**Feature extraction:** `l2_features.py` → `compute_l2_features(sport, l1_features)`

---

### Layer 3 — DK Retail Behavior
**Source:** DraftKings scraper → `snapshots.csv` (every 10 min)
**Philosophy:** DK data is RETAIL — it tells us where the public is, not where the market is going. DK confirms; it never discovers.

**What L3 provides:**
- `bets_pct` / `money_pct` — public bet and money percentages
- `open_line` / `current_line` — DK line movement from open
- `dk_odds` — current American odds
- Timing data (`dk_start_iso`, hours to game)
- Cross-market signals (spread vs ML vs total agreement)

**L3 rules (dk_rules.py):**

| Rule | What It Does | Range |
|------|-------------|-------|
| Market hierarchy weight | SPREAD 1.0, TOTAL 0.9, ML 0.6 | multiplier |
| DK divergence | bets% vs money% gap (threshold: 15, ML: 20) | 0 to +5 |
| DK line move | Confirms sharp move (≥1pt on spreads) | 0 to +3 |
| Timing credibility | Early 0.6, Mid 0.8, Prime 1.0, Late 0.7 | multiplier |
| Retail alignment penalty | bets ≥70% AND money ≥70% → penalize | -5 |
| Parlay distortion | ML money >80% on favorite <-150 | -4 |
| ML-only penalty | ML moved but spread didn't | -3 |
| Cross-market confirm/contradict | Spread+ML agree or disagree | ±2 |
| DK vs sharp alignment | DK matching or conflicting L1 | +3 / -4 |
| Stale price bonus | DK lagging consensus ≥1pt | 0 to +5 |

**Feature extraction:** `dk_rules.py` → `compute_l3_contribution(row)`

---

## 2. LAYER MERGE

**File:** `merge_layers.py` → `merge_all_layers(dk_df, sport)`

The merge joins L1, L2, situational, weather, and sport-specific context data onto the DK DataFrame using a composite key:

**Match key:** `(canonical_key, market, side_normalized)`

Where:
- `canonical_key` = `"{away_norm} @ {home_norm}|{sport}|{date}"` — built by `canonical_match.py`
- `market` = SPREAD / MONEYLINE / TOTAL — inferred from DK's "splits" format
- `side_normalized` = team name normalized (strips line numbers, resolves aliases)

**Team name resolution:** `team_aliases.py` handles DK abbreviations → full names (e.g., "ATL Hawks" → "atlanta hawks", "VGK Golden Knights" → "vegas golden knights"). 61 NBA + NHL aliases, plus NCAAB/NCAAF mappings.

**Layer mode assignment:**

| L1 Available | L2 Available | Mode | Dashboard Label | Score Cap |
|-------------|-------------|------|-----------------|-----------|
| Yes | Yes | L123 | FULL | 100 |
| Yes | No | L13 | PARTIAL | 85 |
| No | Yes | L23 | PARTIAL | 80 |
| No | No | L3_ONLY | LIMITED | 75 |

**Situational data (ESPN):**
- Injuries (home/away lists + counts)
- Rest days + B2B flags (HOME_B2B / AWAY_B2B / BOTH_B2B)
- Pitcher matchup (MLB — ERA, W-L, handedness)
- Probable goalies (NHL — name, status, W-L-OT record)
- NCAAB rankings (AP/NET top 25)

**Weather data (Open-Meteo):** *(v2.1 NEW)*
- Wind speed, temperature, precipitation probability
- Applied to outdoor sports only (NFL, NCAAF, MLB)
- Dome stadiums detected and skipped automatically

**Sport-specific context:** *(v2.1 NEW)*
- MLB: Starting pitcher quality + park factors
- NHL: Goalie confirmation scoring
- NCAAB: Ranking differential

---

## 3. SCORING MODEL (v2.1)

**File:** `scoring_v2.py` → `compute_unified_score(row)`

### Score Formula

```
Score = 50 (base)
  + L1 adjustment       (-5 to +10)
  + L2 adjustment       (-5 to +7)
  + Pattern bonus        (varies by pattern)
  + Cross-market adj     (-2 to +1)
  + Line diff bonus      (0 to +8, currently disabled)
  + Momentum decay       (0 to -3)
  + B2B adjustment       (0 to -1)
  + Injury adjustment    (-2 to +1)        ← v2.1 NEW
  + Weather adjustment   (-3.5 to 0)       ← v2.1 NEW
  + Sport context adj    (-3 to +3)        ← v2.1 NEW
  → Clamped to [floor, cap]
```

**Theoretical range:** 19 to 100 (L123 mode, Pattern A/D/G)

---

### 3A. Layer 1 Adjustment (-5 to +10)

The primary signal driver. L1 is trusted most because sharp books have the best information. BIDIRECTIONAL: penalizes when sharp opposes DK favored side.

**Components:**
| Component | Range | Description |
|-----------|-------|-------------|
| magnitude | 0-1 | Normalized move size |
| agreement_mult | 1.0-2.0 | Both sharp books agree? |
| stability_mult | 0.5-1.2 | Move consistency |
| limit_mult | 1.0-1.2 | Limit size confidence |
| speed_bonus | -4 to +3 | FAST_SNAP: +3 early / -4 late; SLOW_GRIND: -2 |
| key_bonus | 0 or +2 | Crossed key number (3, 7, 10, 14, 17) |
| leader_bonus | 0-1 | Pinnacle leads = +1, Bookmaker.eu = +0.5 |
| DK cross-check | ×0.75 to ×1.15 | Money confirms/contradicts sharps |

---

### 3B. Layer 2 Adjustment (-5 to +7)

Validates or rejects L1 signal using 31-book consensus.

| Agreement | Behavior | Range |
|-----------|----------|-------|
| ≥ 0.6 | Market confirms L1 → positive | 0 to +7 |
| 0.3 - 0.6 | Ambiguous → near-zero | -0.75 to +0.75 |
| ≤ 0.3 | Market rejects L1 → negative | -5 to 0 |

**Trend bonus:** TIGHTENING +1.5, WIDENING -1.5, STABLE 0

---

### 3C. Injury Adjustment (-2 to +1) — v2.1 NEW

**Applies to:** NBA, NHL, NFL only (college rosters too deep to matter)

| Market | Condition | Adjustment |
|--------|-----------|------------|
| ML/SPREAD | Our team 3+ injuries | -2.0 |
| ML/SPREAD | Our team 2+ injuries | -1.0 |
| ML/SPREAD | Opponent 3+ injuries | +1.0 |
| ML/SPREAD | Opponent 2+ injuries | +0.5 |
| TOTAL | Combined 4+ injuries | -1.5 |
| TOTAL | Combined 2+ injuries | -0.5 |

**Source:** ESPN injury lists via `espn_situational.py`

---

### 3D. Weather Adjustment (-3.5 to 0) — v2.1 NEW

**Applies to:** NFL, NCAAF, MLB (outdoor stadiums only)
**Source:** Open-Meteo API (free, no key required), 30-min cache

| Condition | Adjustment | Signal |
|-----------|-----------|--------|
| Wind ≥ 20 mph | -1.5 | HIGH_WIND |
| Wind 15-19 mph | -0.5 | WINDY |
| Precip ≥ 70% or ≥ 2mm | -1.0 | RAIN_LIKELY |
| Precip 40-69% | -0.5 | RAIN_POSSIBLE |
| Temp ≤ 20°F (NFL/NCAAF) | -1.0 | EXTREME_COLD |
| Temp 21-32°F (NFL/NCAAF) | -0.5 | COLD |

**Maximum cumulative:** -3.5 (high wind + rain + extreme cold)
**Dome stadiums:** Automatically detected, return 0 adjustment

**Stadium database:** 32 NFL + 30 MLB venues with (lat, lon, is_dome) in `weather.py`

---

### 3E. Sport-Specific Context Adjustment (-3 to +3) — v2.1 NEW

#### MLB: Pitching + Park Factors

**Starting Pitcher Quality Score** (ERA-based, -3 to +3):

| ERA | Score | Flag |
|-----|-------|------|
| < 2.50 | +3.0 | SP_ACE |
| 2.50-3.00 | +2.0 to +3.0 | SP_ACE |
| 3.00-3.50 | +1.0 to +2.0 | SP_STRONG |
| 3.50-3.75 | +0.5 to +1.0 | SP_STRONG |
| 3.75-4.25 | 0.0 | SP_AVG |
| 4.25-4.75 | -0.5 to -1.0 | SP_WEAK |
| 4.75-5.50 | -1.5 to -2.5 | SP_WEAK |
| > 5.50 | -3.0 | SP_BAD |
| W+L < 3 | -1.0 | SP_UNKNOWN |

**Sample gate:** ERA requires ≥3 decisions (W+L). Below threshold = SP_UNKNOWN with -1.0 penalty.

**MLB adjustment formula:**
- ML/SPREAD: `(our_pitcher_quality - opp_pitcher_quality) × 0.5` — capped at ±3.0
- TOTAL: `avg_pitcher_quality × 0.3` + park adjustment

**Park factors** (static lookup, all 30 MLB teams):

| Park | Factor | Effect on TOTAL |
|------|--------|----------------|
| Coors Field (COL) | 1.28 | +1.5 (strong over lean) |
| Fenway Park (BOS) | 1.10 | +1.5 |
| Yankee Stadium (NYY) | 1.05 | +0.5 |
| Dodger Stadium (LAD) | 0.93 | -0.5 |
| Petco Park (SD) | 0.91 | -1.0 (strong under lean) |
| *All 30 teams mapped* | 0.91-1.28 | varies |

**Source:** `mlb_context.py` — ERA/W/L from ESPN probables, park factors static

#### NHL: Goalie Confirmation

| Status | Starter? | Adjustment | Flag |
|--------|----------|-----------|------|
| Confirmed | Yes (W+L+OT ≥ 20) | +1.0 | G_STARTER_CONFIRMED |
| Confirmed | No (backup) | -2.0 | G_BACKUP_CONFIRMED |
| Probable | Yes | +0.5 | G_PROBABLE |
| Probable | No | -1.0 | G_PROBABLE |
| Unknown | — | 0.0 | G_UNKNOWN |

**Side-relative:** "Our" goalie's quality directly affects adjustment. Opponent's backup gives +50% of absolute value as edge bonus.

**Source:** `nhl_context.py` — goalie status from ESPN NHL scoreboard probables

#### NCAAB: Ranking Differential

| Condition | Adjustment |
|-----------|-----------|
| Rank gap ≥ 15 (we're higher) | +1.0 |
| Rank gap 8-14 (we're higher) | +0.5 |
| Ranked vs unranked (we're ranked) | +0.5 |
| Rank gap ≥ 15 (we're lower) | -1.0 |
| Rank gap 8-14 (we're lower) | -0.5 |
| Unranked vs ranked (we're unranked) | -0.5 |

**Source:** ESPN AP/NET rankings via `espn_situational.py` → `fetch_ncaab_rankings()`

---

## 4. INTERACTION PATTERNS

**Detection order:** F → G → A → D → B → E → C → N (first match wins)

| Pattern | Label | Badge | Conditions | Bonus | Floor | STRONG | Color |
|---------|-------|-------|-----------|-------|-------|--------|-------|
| **F** | LATE_SNAP | SNAP | L1 FAST_SNAP + <1hr to game | -8 | 40 | No | Amber |
| **G** | REVERSE_LINE_MOVE | RLM | bets≥60% + gap≥15% + L1 moved + L2≥0.5 | +4 | 50 | Yes | Purple |
| **A** | SHARP_VS_PUBLIC | SHRP | L1 moved + L2≥0.5 + public heavy | +5 | 50 | Yes | Green |
| **D** | STALE_PRICE | STALE | L1 moved + L2≥0.5 + DK stale | +4 | 50 | Yes | Red |
| **B** | RETAIL_ALIGNMENT | ALGN | L1 moved + L2≥0.5 + public NOT heavy | cap 70 | 45 | No | Blue |
| **E** | CONSENSUS_REJECTS | REJ | L1 moved + L2<0.3 | -6, cap 65 | 40 | No | Orange |
| **C** | RETAIL_ALIGNMENT | PUB | No L1 move + public heavy | -5 | 40 | No | Gold |
| **N** | NEUTRAL | NEU | No strong pattern | 0 | 40 | No | Gray |

### Pattern G — Reverse Line Movement

The strongest cross-layer signal. Fires when public bets are heavy on one side but money and sharp lines move the opposite direction.

**RLM Conditions (all must pass):**
1. `bets_pct ≥ 60` — public heavily on this side
2. `bets_pct - money_pct ≥ 15` — money NOT following bets
3. `l1_move_dir ≠ 0` — sharp books moved
4. `l2_consensus_agreement ≥ 0.5` — consensus confirms sharp direction

**RLM Strength (0 to 1):**
```
gap_factor = min(bets_money_gap / 30.0, 1.0)
strength = gap_factor × max(l1_sharp_strength, 0.3) × max(l2_agreement, 0.5)
```

---

## 5. CROSS-MARKET & DECAY

### Spread-Total Cross Check (-2 to +1)
- Spread positive + Total positive = consistent (+1)
- Spread positive + Total negative = contradiction (-2)

### Momentum Decay (0 to -3)
If score hasn't increased in 4+ ticks: -1 per tick over 3, max -3. Prevents stale persistent signals.

### B2B Adjustment (0 to -1)
HOME_B2B or AWAY_B2B → -1.0 slight penalty.

---

## 6. SCORE CAPS AND FLOORS

### Layer Mode Caps — REMOVED (v2.1)

Layer-based caps were removed in v2.1. All layer modes now use the universal cap of 100. Layers contribute to the score through adjustments — more layers = more adjustment components that can fire — but they no longer artificially limit the ceiling. An L3_ONLY row with strong DK signals can theoretically reach 100, though in practice it rarely exceeds ~70 without L1/L2 confirmation.

| Mode | Cap | Who Gets It |
|------|-----|-------------|
| L123 | 100 | Sharp + Consensus + DK (best data) |
| L13 | 100 | Sharp + DK (no consensus validation) |
| L23 | 100 | Consensus + DK (no sharp signal) |
| L3_ONLY | 100 | DK only (retail-only, limited upside in practice) |

### Pattern Floors

| Pattern | Floor | Why |
|---------|-------|-----|
| A, D, G | 50 | Strong signal patterns never go below neutral |
| B | 45 | Aligned but priced in |
| C, E, F, N | 40 | Weaker or warning patterns |
| L3_ONLY | 0 | No floor for DK-only |

### Pattern Caps
- Pattern B: capped at 70 (aligned = priced in, never Strong)
- Pattern E: capped at 65 (consensus rejects = limited upside)

---

## 7. DECISION THRESHOLDS

| Decision | Score | Net Edge (SPREAD/ML) | Net Edge (TOTAL) | STRONG Eligible |
|----------|-------|---------------------|-------------------|-----------------|
| **STRONG_BET** | ≥ 70 | ≥ 10 | ≥ 12 | True |
| **BET** | ≥ 67 | ≥ 10 | ≥ 12 | — |
| **LEAN** | ≥ 60 | any | any | — |
| **NO BET** | < 60 | any | any | — |

### STRONG_BET Eligibility — 3 Paths (v2.1)

STRONG_BET can be awarded through 3 independent paths. All paths share common gates (score, edge, persistence), but differ in how eligibility is established.

**Common gates (all paths):**

| Gate | Condition |
|------|-----------|
| Score | ≥ 70 |
| Net Edge | ≥ 10 (SPREAD/ML) or ≥ 12 (TOTAL) |
| Persistence | strong_streak ≥ 2 |
| Layer Mode | NOT L3_ONLY (v2.1 guard — L3_ONLY can never produce STRONG) |

**Path 1: Pattern-driven** (most common)
- Pattern A, D, or G (strong signal patterns)
- Pattern's `strong_eligible` flag is True

**Path 2: Sharp Certified FULL**
- `sharp_cert_tier` = "FULL" (L1 sharp books strongly confirm)
- Does not require specific pattern

**Path 3: Score-only** (higher bar)
- Score ≥ 75 (STRONG_SCORE_ONLY_MIN)
- Net Edge ≥ 12 (STRONG_SCORE_ONLY_EDGE)
- strong_streak ≥ 3 (STRONG_SCORE_ONLY_PERSIST)

**Additional gates in main.py _is_strong_eligible():**

| Gate | Condition |
|------|-----------|
| Timing | ≠ LATE |
| Market Read | ≠ "Public Drift" |
| Cross-Market | No contradiction |
| Stability | last_score within delta of peak (NCAAB: 2pts, others: 3pts) |
| Early Block | NCAAB/NCAAF: timing ≠ EARLY |

**L3_ONLY STRONG guard:** Encoded directly in `scoring_v2.compute_unified_score()` — L3_ONLY rows always have `strong_eligible=False` regardless of score or edge. This is tested in `test_invariants.py::test_l3_only_no_strong`.

---

## 8. CLV TRACKING — v2.1 NEW

### What is CLV?
**Closing Line Value** measures whether the engine consistently gets better numbers than the market closing line. It's the gold standard for betting engine validation — win rate has high variance, but sustained positive CLV proves genuine edge.

### How It Works

**Decision Line:** Captured at freeze time — the DK line when STRONG_BET/BET is first awarded. The freeze ledger's dedup logic (STRONG > BET > LEAN > NO BET) ensures the decision line reflects the peak conviction moment.

**Closing Line:** Last DK snapshot before `dk_start_iso` per game/side. Uses the existing 10-min scrape cadence. Captured by `capture_closing_lines()` in main.py.

**CLV Calculation:**

| Market | Formula | Positive Means |
|--------|---------|----------------|
| SPREAD | `decision_val - closing_val` | Got better number (e.g., -3 when market closed at -3.5) |
| TOTAL OVER | `closing - decision` | Took over at lower number than close |
| TOTAL UNDER | `decision - closing` | Took under at higher number than close |
| MONEYLINE | `closing_implied_prob - decision_implied_prob` | Got better odds |

**CLV Columns in results_resolved.csv:**
- `decision_line` — DK line at decision time
- `decision_line_val` — numeric value extracted
- `decision_odds` — DK odds at decision time
- `closing_line` — last DK line before game start
- `closing_line_val` — numeric value extracted
- `clv` — closing line value (positive = engine beat the close)
- `clv_direction` — BEAT_CLOSE / LOST_CLOSE / PUSH

**CLV in KPIs (kpi_builder.py):**
- Average CLV by decision type, sport, market
- CLV buckets with win rates: ≥2.0 / 1.0-1.9 / 0.1-0.9 / 0 / negative
- Forward-only: only applies to decisions made after CLV deployment

---

## 9. THE PIPELINE (Execution Order)

```
1.  DK snapshot ingest (snapshots.csv)
2.  Odds API pull: L1 sharp (OddsPapi) + L2 consensus (The-Odds-API)
3.  Feature extraction: l1_features, l2_features
4.  Layer merge: merge_all_layers() — joins L1/L2/situational/weather/sport-context
5.  Market read computation (v1.2 logic, retained for L3-only rows)
6.  3-Layer scoring: scoring_v2.compute_unified_score() for L123/L13/L23 rows
7.  DK-only scoring: v1.2 side-level scoring for L3_ONLY rows
8.  Post-scoring passes (persistence cap, SPREAD dampening, ML→Spread reinforcement)
9.  Metrics tap (row_state.csv, signal_ledger.csv)
10. Aggregation (game-level: net_edge, favored_side, decision)
11. Strong eligibility join
12. Dashboard write (dashboard.csv, dashboard.html)
13. Freeze ledger write (decision_freeze_ledger.csv — includes decision_line for CLV)
14. Closing line capture (last DK snapshot before game start)
15. Outcomes enrichment + grading (results_resolved.csv — includes CLV)
16. KPI layer (read-only, includes CLV analysis)
```

**Dual scoring paths:**
- Rows with L1 and/or L2 data → `scoring_v2.py` (3-layer model)
- Rows with DK only → existing v1.2 scoring in `main.py` (preserved)
- v2.0+ score becomes PRIMARY (`confidence_score`) when available
- v1.2 score used as fallback for L3_ONLY rows

---

## 10. FILES AND THEIR ROLES

### Data Files

| File | Role | Writer |
|------|------|--------|
| `snapshots.csv` | Raw DK market feed (append-only) | DK scraper |
| `l1_sharp.csv` | Sharp book lines (Pinnacle, Bookmaker.eu) | l1_scraper.py |
| `l1_open_registry.csv` | Opening lines for L1 move detection | l1_scraper.py |
| `l2_consensus.csv` | Raw 31-book consensus lines | l2_scraper.py |
| `l2_consensus_agg.csv` | Aggregated consensus (median, std, n_books per side) | l2_scraper.py |
| `dashboard.csv` | Game-level aggregated summary for UI | aggregation |
| `dashboard.html` | Engine-generated HTML report | build_dashboard |
| `decision_snapshots.csv` | Ephemeral per-run side-level view (pre-freeze) | build_dashboard |
| `decision_freeze_ledger.csv` | Canonical historical decisions (append-only, never downgrade) | freeze step |
| `results_resolved.csv` | Post-game graded results + frozen decisions + CLV | outcomes |
| `final_scores_history.csv` | ESPN final scores | outcomes |
| `row_state.csv` | Side-level lifecycle memory (score, peak, streak) | metrics tap |
| `signal_ledger.csv` | Event log of threshold crossings | metrics tap |

### Code Modules

| File | Role |
|------|------|
| `main.py` | Core pipeline (~5900 lines, LFS tracked). Dashboard build, outcomes, CLV, freeze ledger. |
| `scoring_v2.py` | Unified scoring: dk_base + 10 adjustment components, pattern detection, floors/caps |
| `dk_scoring.py` | DK base score (16 components + book response + line trajectory) |
| `l1_features.py` | L1 sharp book feature extraction |
| `l2_features.py` | L2 consensus feature extraction |
| `dk_rules.py` | DK retail interpretation rules, L3 contribution |
| `merge_layers.py` | Layer merge: joins L1/L2/ESPN/weather/sport-context onto DK DataFrame |
| `engine_config.py` | All thresholds, weights, API config, season calendar |
| `team_aliases.py` | Team name normalization + DK/API alias resolution |
| `canonical_match.py` | Cross-source game matching (canonical key builder) |
| `odds_api.py` | The-Odds-API wrapper (L2 consensus data) |
| `l1_scraper.py` | OddsPapi wrapper (L1 sharp data) |
| `l2_scraper.py` | Consensus aggregation from The-Odds-API |
| `espn_situational.py` | ESPN: injuries, rest days, pitcher matchups, goalie status, NCAAB rankings |
| `weather.py` | Open-Meteo weather API + 62 stadium database (NFL + MLB) |
| `mlb_context.py` | SP quality scoring (ERA-based), park factors (30 teams) |
| `nhl_context.py` | Goalie confirmation scoring, starter/backup detection |
| `kpi_builder.py` | KPI analytics + CLV analysis |
| `serve.py` | HTTP server, routes /board.html to live dashboard |
| `site/board.html` | Live dashboard UI (fetches CSVs, renders board) |

---

## 11. API INFRASTRUCTURE

### OddsPapi (L1 Sharp)
- **Budget:** 250 req/month
- **Max 3 sports per pull** (top priority by month)
- **Tournament IDs:** NBA=132, NCAAB=648 (NHL/MLB/NFL/UFC: TBD)
- **Sharp books available:** Pinnacle, Singbet, SBOBet, BetCris, Circa Sports, Bookmaker.eu
- **Currently returning:** Pinnacle + Bookmaker.eu (2 of 6)

### The-Odds-API (L2 Consensus)
- **Budget:** 500 req/month free tier
- **31 books** across US + EU regions
- **Budget guard:** Auto-skip if < 30 requests remain

### ESPN (Situational)
- Injuries, rest days, B2B detection
- Final scores for outcome grading
- Probable pitchers (MLB): ERA, W-L, handedness
- Probable goalies (NHL): name, status, W-L-OT record
- Rankings (NCAAB): AP/NET top 25
- Cache TTL: 30 minutes

### Open-Meteo (Weather) — v2.1 NEW
- **Completely free**, no API key required
- Hourly forecast: wind, temperature, precipitation probability
- 30-minute disk cache
- 62 stadium locations (32 NFL + 30 MLB)

### DraftKings (L3 Retail)
- Scraped every 10 minutes via `run_all_sports.sh`
- Public bets%, money%, lines, odds
- All active sports pulled every run

---

## 12. DASHBOARD (dashboard.html)

### Game Cell Badges

Each game row shows contextual badges after the game name:

| Badge | Sport | Color | Example |
|-------|-------|-------|---------|
| B2B | NBA/NHL | Red | `B2B` |
| Injuries | NBA/NHL/NFL | Red | `H:3inj A:1inj` |
| Weather | NFL/NCAAF/MLB | Blue/Sun | `🌬 HIGH_WIND` or `☀` |
| Dome | NFL/MLB | Gray | `🏟` |
| Pitcher | MLB | Green/Gray/Red | `RHP Cole 3.21` |
| Park Factor | MLB | Gold/Blue | `⚾ 1.28x` |
| Goalie | NHL | Green/Yellow/Gray | `✅ Shesterkin` |
| Ranking | NCAAB | Purple | `#5 vs #22` |

### v2.1 Dashboard Additions
- **Pitcher badges** — last name + ERA, colored by quality (green=ace/strong, gray=avg, red=weak/bad)
- **Park factor badges** — shown for extreme parks (≥1.05 or ≤0.95), gold=hitter, blue=pitcher
- **Goalie badges** — confirmed (green ✅), probable (yellow ❓), unknown (gray ❓)
- **Ranking badges** — purple badges showing our rank and opponent rank
- **Weather badges** — wind/rain/cold indicators with hover tooltips showing exact conditions

### KPIs
- By Decision: STRONG_BET / BET / LEAN / NO BET win rates
- By Confidence Bucket: 75+ / 70-74 / 65-69 / 60-64 / <60
- By Net Edge: 13+ / 9-12 / 5-8 / 0-4
- By Market Read: Freeze Pressure / Stealth Move / etc.
- By Sport
- By Market Type (SPREAD / TOTAL / MONEYLINE)
- Fav vs Dog (SPREAD only)
- **By CLV bucket** *(v2.1 NEW)*: ≥2.0 / 1.0-1.9 / 0.1-0.9 / 0 / negative — with win rates

**KPI epoch:** Reset to 2026-03-05 (v2.0 clean baseline). Only counts the picked side (favored_side), not both sides.

---

## 13. DEPLOYMENT

- **Server:** 159.65.167.146 (port 2222)
- **Path:** /opt/red-fox-market-dynamics
- **Auto-deploy:** `git pull origin main` every minute via cron
- **DK scraper:** `run_all_sports.sh` every 10 minutes
- **Odds pulls:** 3x daily (11:30, 15:30, 18:30 ET)
- **Python:** .venv/bin/python

---

## 14. SPORT SEASON CALENDAR

| Sport | Season | L1 Tournament ID |
|-------|--------|------------------|
| NBA | Oct - Jun | 132 |
| NHL | Oct - Jun | TBD |
| MLB | Mar - Nov | TBD |
| NFL | Sep - Feb | TBD |
| NCAAB | Nov - Apr | 648 |
| NCAAF | Aug - Jan | TBD |
| UFC | Year-round | TBD |

---

## 15. SINGLE WRITER RULE (Enforced)

| File | Writer |
|------|--------|
| snapshots.csv | DK scraper |
| l1_sharp.csv | l1_scraper.py |
| l2_consensus.csv / l2_consensus_agg.csv | l2_scraper.py |
| dashboard.csv | aggregation step |
| row_state.csv | metrics tap |
| signal_ledger.csv | metrics tap |
| final_scores_history.csv | outcomes |
| results_resolved.csv | outcomes |
| decision_freeze_ledger.csv | freeze step |

No duplicate writers allowed.

---

## 16. KEY ENGINE LAWS (Non-Negotiable)

1. **L1 leads.** Sharp books provide the primary signal. DK confirms, never discovers.
2. **L2 validates.** Consensus either confirms or rejects L1. This determines confidence.
3. **L3 modifies.** DK retail behavior adjusts score within bounds, never overrides L1/L2.
4. **Patterns classify.** Every scored row gets exactly one pattern (A-G or N). Patterns control floors, caps, and STRONG eligibility.
5. **Layers contribute, not limit.** More layers = more adjustment components. No layer-based caps (removed v2.1). L3_ONLY rows have limited upside in practice, not by artificial cap.
6. **STRONG requires confirmation.** STRONG_BET has 3 paths (pattern, sharp certified, score-only) but L3_ONLY can NEVER produce STRONG — enforced in code.
7. **Scoring happens once.** No downstream layer recomputes scores.
8. **Freeze is permanent.** Decisions freeze at dashboard build time; never downgraded.
9. **Outcomes grade only.** Post-game results do not influence scoring or decisions.
10. **KPIs analyze only.** Read-only layer. Counts only the picked side.
11. **Context adjusts, never overrides.** Weather, injuries, and sport context modify the score within bounds but cannot override L1/L2 signals. *(v2.1 NEW)*

---

## 17. SCORING CALIBRATION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-02-27 | Initial DK-only model. Color classification, additive signals, fixed base 50. |
| v1.1 | 2026-03-01 | STRONG_BET wired, canonical keys, elig_map, SPREAD dampening (×0.92, +0.55 above 70), persistence cap (+6/tick), ML risk governor, key_number +6 SPREAD. |
| v1.1 Patch 9 | 2026-03-01 | SPREAD elevated (lm ×3.0, key_number +6 all sports), TOTAL suppressed (D ×0.3, net_edge min 12), Reverse Pressure +8, Neutral -2, Contradiction 0. |
| v1.2 Phase A | 2026-03-02 | Regime classifier (A/B/C/D/N), v1.2 RLM signal (bets-based), combined divergence multiplier (5 components), dynamic base (44/50/52), Contradiction -4, EARLY timing 0, market_read persistence, dashboard UI badges. |
| v1.2 Phase B | 2026-03-02 | Sport-relative longshot penalty (ML only, per-sport baselines). |
| v1.2 Rebalance | 2026-03-02 | NCAAB -4 removed (redundant), NCAAF -2 added, NHL puck line governor -3, decision thresholds lowered (LEAN 60, BET 67, STRONG 70). |
| v2.0 | 2026-03-04 | 3-Layer architecture. L1 sharp (OddsPapi), L2 consensus (The-Odds-API), L3 DK (retail). New modules: scoring_v2.py, dk_rules.py, l1_features.py, l2_features.py, merge_layers.py. Pattern detection (A-G). Cross-layer RLM (Pattern G). Layer mode caps (100/85/80/75). STRONG requires L123 + Pattern A/D/G. DK demoted to modifier (-10 to +10). 61 NBA/NHL team aliases. Dashboard pattern + layer badges. |
| v2.0.1 | 2026-03-05 | Scoring signal overhaul: book response philosophy (read the BOOK not bettors), continuous divergence signals, ML vs Spread cross-check, line trajectory tracking. |
| **v2.1** | **2026-03-05** | **CLV tracking. ESPN injury + weather + sport context. Scoring recalibration: effective_move_mag (juice/odds movement detection), boosted confirmed signals, 2-tier line movement multipliers, line trajectory bonuses doubled. Silent failure logging. L3_ONLY STRONG guard. Score decomposition columns (dk_base, pattern_bonus, decay). 5 invariant tests. Tick shadowing fix.** |

---

## 18. FUTURE (Phase 7 — Needs 100+ Graded Decisions)

| Feature | Description | Prerequisite |
|---------|-------------|-------------|
| Score Calibration | Map raw scores to actual win probabilities | 100+ graded BET/STRONG decisions |
| Kelly Criterion Sizing | Optimal bet sizing from calibrated probabilities | Calibration complete |
| Bayesian Score Updates | Prior from historical patterns, posterior from live signals | Sufficient historical data |

These are deferred until the engine has enough resolved decisions to build reliable calibration curves.

---

*Engine v2.1 is the first version with full situational intelligence — weather, injuries, pitching, goalies, and rankings all flow through scoring alongside the 3-layer signal model. CLV tracking enables rigorous engine validation beyond win/loss noise.*
