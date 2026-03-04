# Red Fox Market Intelligence — Engine Architecture v2.0
*Updated: 2026-03-04 — 3-Layer Scoring Model, Pattern Detection, RLM, complete redesign from v1.x*

---

## EVOLUTION SUMMARY

| Version | Date | Architecture |
|---------|------|-------------|
| v1.0 | 2026-02-27 | Single-layer DK scoring. Color classification, additive signals, fixed base 50. |
| v1.1 | 2026-03-01 | STRONG_BET wiring, canonical keys, elig_map, persistence cap, SPREAD dampening. Still single-layer. |
| v1.2 | 2026-03-02 | Regime classifier, RLM signal, combined divergence multiplier, dynamic base, sport-specific rebalancing. Still single-layer. |
| **v2.0** | **2026-03-04** | **3-Layer model. L1 (sharp books via OddsPapi), L2 (31-book consensus via The-Odds-API), L3 (DK retail behavior). Pattern detection system. Cross-layer interaction scoring. Layer mode caps. Full redesign.** |

**What changed from v1.x → v2.0:**
- v1.x scored everything from DK data alone (bets%, money%, line movement)
- v2.0 uses DK as one of three independent data layers, each with defined contribution ranges
- Scoring logic moved from inline main.py to modular files: `scoring_v2.py`, `dk_rules.py`, `l1_features.py`, `l2_features.py`, `merge_layers.py`
- Pattern detection replaces simple market_read as primary signal classifier
- Layer mode (FULL/PARTIAL/LIMITED) caps scores based on data availability

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

The merge joins L1, L2, and situational data onto the DK DataFrame using a composite key:

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
- Pitcher matchup (MLB)

---

## 3. SCORING MODEL (v2.0)

**File:** `scoring_v2.py` → `compute_3layer_score(row)`

### Score Formula

```
Score = 50 (base)
  + L1 contribution    (0 to +18)
  + L2 contribution    (-8 to +10)
  + L3 contribution    (-10 to +10)
  + Pattern bonus      (varies by pattern)
  + Cross-market adj   (-2 to +1)
  + Momentum decay     (0 to -3)
  + B2B adjustment     (0 to -1)
  → Clamped to [floor, cap]
```

**Theoretical range:** 19 to 100 (L123 mode, Pattern A/D/G)

---

### 3A. Layer 1 Contribution (0 to +18)

The primary signal driver. L1 is trusted most because sharp books have the best information.

**Formula:**
```
raw = (magnitude × agreement_mult × stability_mult × limit_mult × 18) + speed_bonus + key_bonus
clamped to [0, 18]
```

| Component | Range | Description |
|-----------|-------|-------------|
| magnitude | 0-1 | Normalized move size |
| agreement_mult | 1.0-2.0 | Both sharp books agree? |
| stability_mult | 0.5-1.2 | Move consistency |
| limit_mult | 1.0-1.2 | Limit size confidence |
| speed_bonus | -4 to +3 | FAST_SNAP: +3 early / -4 late; SLOW_GRIND: -2 |
| key_bonus | 0 or +2 | Crossed key number (3, 7, 10, 14, 17) |

---

### 3B. Layer 2 Contribution (-8 to +10)

Validates or rejects L1 signal using 31-book consensus.

| Agreement | Behavior | Range |
|-----------|----------|-------|
| ≥ 0.6 | Market confirms L1 → positive | 0 to +10 |
| 0.3 - 0.6 | Ambiguous → near-zero | -0.75 to +0.75 |
| ≤ 0.3 | Market rejects L1 → negative | -8 to 0 |

**Trend bonus:** TIGHTENING +1.5, WIDENING -1.5, STABLE 0

**Validation strength** (composite 0-1):
- Book count (15%): more books = stronger
- Dispersion (30%): tighter = stronger
- Trend (15%): tightening = confirming
- Agreement (30%): books matching L1 = strongest
- Pinnacle proximity (10%): small Pinn gap = aligned

---

### 3C. Layer 3 Contribution (-10 to +10)

DK retail behavior as modifier (confirms but never discovers).

See Section 1 Layer 3 rules table. Composite of divergence score, line move, timing credibility, and all penalty/bonus rules, clamped to [-10, +10].

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

### Pattern G — Reverse Line Movement (v2.0 NEW)

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

### Layer Mode Caps

| Mode | Cap | Who Gets It |
|------|-----|-------------|
| L123 | 100 | Sharp + Consensus + DK (best data) |
| L13 | 85 | Sharp + DK (no consensus validation) |
| L23 | 80 | Consensus + DK (no sharp signal) |
| L3_ONLY | 75 | DK only (v1.2 behavior, retail-only) |

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

### STRONG_BET Eligibility (all gates must pass)

| Gate | Condition |
|------|-----------|
| Layer Mode | L123 only (full 3-layer) |
| Pattern | A, D, or G only |
| Score | ≥ 70 |
| Net Edge | ≥ 10 (SPREAD/ML) or ≥ 12 (TOTAL) |
| Timing | ≠ LATE |
| Market Read | ≠ "Public Drift" |
| Cross-Market | No contradiction |
| Persistence | strong_streak ≥ 2 (NCAAB: ≥ 3) |
| Stability | last_score within delta of peak (NCAAB: 2pts, others: 3pts) |
| Early Block | NCAAB/NCAAF: timing ≠ EARLY |

**v2.0 change:** STRONG_BET requires L123 mode — cannot be awarded without both sharp and consensus data confirming.

---

## 8. THE PIPELINE (Execution Order)

```
1. DK snapshot ingest (snapshots.csv)
2. Odds API pull: L1 sharp (OddsPapi) + L2 consensus (The-Odds-API)
3. Feature extraction: l1_features, l2_features
4. Layer merge: merge_all_layers() — joins L1/L2/situational onto DK rows
5. Market read computation (v1.2 logic, retained for L3-only rows)
6. 3-Layer scoring: scoring_v2.compute_3layer_score() for L123/L13/L23 rows
7. DK-only scoring: v1.2 side-level scoring for L3_ONLY rows
8. Post-scoring passes (persistence cap, SPREAD dampening, ML→Spread reinforcement)
9. Metrics tap (row_state.csv, signal_ledger.csv)
10. Aggregation (game-level: net_edge, favored_side, decision)
11. Strong eligibility join
12. Dashboard write (dashboard.csv)
13. Freeze ledger write (decision_freeze_ledger.csv)
14. Outcomes enrichment + grading (results_resolved.csv)
15. KPI layer (read-only)
```

**Dual scoring paths:**
- Rows with L1 and/or L2 data → `scoring_v2.py` (3-layer model)
- Rows with DK only → existing v1.2 scoring in `main.py` (preserved)
- v2.0 score becomes PRIMARY (`confidence_score`) when available
- v1.2 score used as fallback for L3_ONLY rows

---

## 9. FILES AND THEIR ROLES

### Data Files

| File | Role | Writer |
|------|------|--------|
| `snapshots.csv` | Raw DK market feed (append-only) | DK scraper |
| `l1_sharp.csv` | Sharp book lines (Pinnacle, Bookmaker.eu) | l1_scraper.py |
| `l1_open_registry.csv` | Opening lines for L1 move detection | l1_scraper.py |
| `l2_consensus.csv` | Raw 31-book consensus lines | l2_scraper.py |
| `l2_consensus_agg.csv` | Aggregated consensus (median, std, n_books per side) | l2_scraper.py |
| `dashboard.csv` | Game-level aggregated summary for UI | aggregation |
| `dashboard.html` | Engine-generated HTML report (legacy) | build_dashboard |
| `decision_snapshots.csv` | Ephemeral per-run side-level view (pre-freeze) | build_dashboard |
| `decision_freeze_ledger.csv` | Canonical historical decisions (append-only, never downgrade) | freeze step |
| `results_resolved.csv` | Post-game graded results + frozen decisions | outcomes |
| `final_scores_history.csv` | ESPN final scores | outcomes |
| `row_state.csv` | Side-level lifecycle memory (score, peak, streak) | metrics tap |
| `signal_ledger.csv` | Event log of threshold crossings | metrics tap |

### Code Modules

| File | Role |
|------|------|
| `main.py` | Core pipeline, v1.2 scoring (L3-only fallback), dashboard build, outcomes |
| `scoring_v2.py` | 3-layer scoring model, pattern detection, score floors/caps |
| `dk_rules.py` | DK retail interpretation rules, L3 contribution |
| `l1_features.py` | L1 sharp book feature extraction |
| `l2_features.py` | L2 consensus feature extraction |
| `merge_layers.py` | Layer merge: joins L1/L2/situational onto DK DataFrame |
| `engine_config.py` | All thresholds, weights, API config, season calendar |
| `team_aliases.py` | Team name normalization + DK/API alias resolution |
| `canonical_match.py` | Cross-source game matching (canonical key builder) |
| `odds_api.py` | The-Odds-API wrapper (L2 consensus data) |
| `l1_scraper.py` | OddsPapi wrapper (L1 sharp data) |
| `l2_scraper.py` | Consensus aggregation from The-Odds-API |
| `espn_situational.py` | ESPN injuries, rest days, pitcher matchups |
| `serve.py` | HTTP server, routes /board.html to live dashboard |
| `site/board.html` | Live dashboard UI (fetches CSVs, renders board) |

---

## 10. API INFRASTRUCTURE

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
- Cache TTL: 30 minutes

### DraftKings (L3 Retail)
- Scraped every 10 minutes via `run_all_sports.sh`
- Public bets%, money%, lines, odds
- All active sports pulled every run

---

## 11. DASHBOARD (board.html)

### Board Columns
Rank | Game | Market | Bets/Money | Open | Current | Edge | Conf | Bucket | **Pattern** | **Layer** | Timing | Mkt Read | Regime | Decision | Play

### v2.0 Dashboard Additions
- **Pattern column** — colored badges: RLM (purple), SHRP (green), STALE (red), ALGN (blue), SNAP (amber), PUB (gold), REJ (orange), NEU (gray)
- **Layer column** — FULL (green), PARTIAL (amber/blue), LIMITED (gray) with tooltips showing which layers are present
- **Confidence** — uses v2_score when available, falls back to v1.2 game_confidence for L3-only

### KPIs
- By Decision: STRONG_BET / BET / LEAN / NO BET win rates
- By Confidence Bucket: 75+ / 70-74 / 65-69 / 60-64 / <60
- By Net Edge: 13+ / 9-12 / 5-8 / 0-4
- By Market Read: Freeze Pressure / Stealth Move / etc.
- By Sport
- By Market Type (SPREAD / TOTAL / MONEYLINE)
- Fav vs Dog (SPREAD only)

**KPI epoch:** v2.0 resets to 2026-03-05 (fresh start). Only counts the picked side (favored_side), not both sides.

---

## 12. DEPLOYMENT

- **Server:** 159.65.167.146 (port 2222)
- **Path:** /opt/red-fox-market-dynamics
- **Auto-deploy:** `git pull origin main` every minute via cron
- **DK scraper:** `run_all_sports.sh` every 10 minutes
- **Odds pulls:** 3x daily (11:30, 15:30, 18:30 ET)
- **Python:** .venv/bin/python

---

## 13. SPORT SEASON CALENDAR

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

## 14. SINGLE WRITER RULE (Enforced)

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

## 15. KEY ENGINE LAWS (Non-Negotiable)

1. **L1 leads.** Sharp books provide the primary signal. DK confirms, never discovers.
2. **L2 validates.** Consensus either confirms or rejects L1. This determines confidence.
3. **L3 modifies.** DK retail behavior adjusts score within bounds, never overrides L1/L2.
4. **Patterns classify.** Every scored row gets exactly one pattern (A-G or N). Patterns control floors, caps, and STRONG eligibility.
5. **Layer mode caps.** More data = higher ceiling. L3-only rows cannot exceed 75.
6. **STRONG requires FULL.** STRONG_BET needs all 3 layers (L123) + Pattern A/D/G.
7. **Scoring happens once.** No downstream layer recomputes scores.
8. **Freeze is permanent.** Decisions freeze at dashboard build time; never downgraded.
9. **Outcomes grade only.** Post-game results do not influence scoring or decisions.
10. **KPIs analyze only.** Read-only layer. Counts only the picked side.

---

## 16. SCORING CALIBRATION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-02-27 | Initial DK-only model. Color classification, additive signals, fixed base 50. |
| v1.1 | 2026-03-01 | STRONG_BET wired, canonical keys, elig_map, SPREAD dampening (×0.92, +0.55 above 70), persistence cap (+6/tick), ML risk governor, key_number +6 SPREAD. |
| v1.1 Patch 9 | 2026-03-01 | SPREAD elevated (lm ×3.0, key_number +6 all sports), TOTAL suppressed (D ×0.3, net_edge min 12), Reverse Pressure +8, Neutral -2, Contradiction 0. |
| v1.2 Phase A | 2026-03-02 | Regime classifier (A/B/C/D/N), v1.2 RLM signal (bets-based), combined divergence multiplier (5 components), dynamic base (44/50/52), Contradiction -4, EARLY timing 0, market_read persistence, dashboard UI badges. |
| v1.2 Phase B | 2026-03-02 | Sport-relative longshot penalty (ML only, per-sport baselines). |
| v1.2 Rebalance | 2026-03-02 | NCAAB -4 removed (redundant), NCAAF -2 added, NHL puck line governor -3, decision thresholds lowered (LEAN 60, BET 67, STRONG 70). |
| **v2.0** | **2026-03-04** | **3-Layer architecture. L1 sharp (OddsPapi), L2 consensus (The-Odds-API), L3 DK (retail). New modules: scoring_v2.py, dk_rules.py, l1_features.py, l2_features.py, merge_layers.py. Pattern detection (A-G). Cross-layer RLM (Pattern G). Layer mode caps (100/85/80/75). STRONG requires L123 + Pattern A/D/G. DK demoted to modifier (-10 to +10). 61 NBA/NHL team aliases. Dashboard pattern + layer badges.** |

---

*Engine v2.0 is the first version where all three data layers are active in production. Previous versions scored exclusively from DK retail data.*
