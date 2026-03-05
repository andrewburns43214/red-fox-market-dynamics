# Red Fox Market Intelligence — Complete System Reference
*Master document covering the full system from v1.0 to v2.1*
*Last updated: 2026-03-05*

---

## WHAT IS RED FOX?

Red Fox is a sports betting market intelligence engine. It ingests data from three independent sources — sharp bookmakers, 31-book market consensus, and DraftKings retail behavior — then scores every available bet on a 0-100 scale. The engine runs autonomously on a cloud server, scraping DK every 10 minutes and pulling odds data 3x daily, producing a live dashboard with scored picks.

**The core insight:** DraftKings is a retail sportsbook. The money there comes from recreational bettors and whales, not sharps. DK's own LINE RESPONSE to that money — not the money itself — is the real signal. When Pinnacle (the sharpest book in the world) moves a line and DK hasn't caught up, that's exploitable edge.

**What it outputs:**
- Live dashboard with every active game scored
- Decision labels: STRONG_BET / BET / LEAN / NO BET
- Pattern classification: what TYPE of edge exists (sharp vs public, stale price, RLM, etc.)
- Contextual data: weather, injuries, pitcher matchups, goalie status, rankings
- Post-game grading: did the pick win? Did we beat the closing line?

---

## THE DATA FLOW (End to End)

```
                        INGESTION                    MERGE                     SCORING                    OUTPUT
                    ________________            ________________           ________________           ________________
                   |                |          |                |         |                |         |                |
  DraftKings  ---> | snapshots.csv  |--+       |                |         |                |         | dashboard.csv  |
  (every 10 min)   |________________|  |       |                |         |  scoring_v2.py |         | dashboard.html |
                                       +-----> | merge_layers.py|-------> |  (11 scoring   |-------> | freeze_ledger  |
  Pinnacle/Sharp-> | l1_sharp.csv   |--+       |                |         |   components)  |         | row_state.csv  |
  (3x daily)       |________________|  |       |                |         |                |         |________________|
                                       +-----> |  + weather.py  |         |  dk_scoring.py |
  31-Book Consens->| l2_consensus   |--+       |  + mlb_context |         |  (16 DK base   |              |
  (3x daily)       |________________|  |       |  + nhl_context |         |   components)  |              v
                                       +-----> |  + ESPN situa. |         |________________|         ________________
  ESPN (free)  --->| injuries, B2B  |--+       |________________|                                   |                |
                   | pitchers,goalies|                                                               | results_resolved
                   | rankings       |                                                               | (post-game     |
                   |________________|                                                               |  grading + CLV)|
                                                                                                    |________________|
  Open-Meteo  ---->| weather cache  |
  (free, no key)   |________________|
```

---

## PART 1: DATA SOURCES

### 1A. DraftKings (Layer 3 — Retail)

**What:** Public betting data scraped from DK's website
**Frequency:** Every 10 minutes via `run_all_sports.sh`
**File:** `data/snapshots.csv`

**Fields captured per side:**
| Field | Example | What It Tells You |
|-------|---------|-------------------|
| `bets_pct` | 72% | What percentage of TICKETS are on this side |
| `money_pct` | 45% | What percentage of DOLLARS are on this side |
| `open_line` | -3.5 | Where DK opened the line |
| `current_line` | -4.0 | Where DK's line is now |
| `current_odds` | -110 | Current American odds |
| `dk_start_iso` | 2026-03-05T23:00:00Z | Game start time |

**Why bets vs money matters:** If 72% of tickets are on Team A but only 45% of the money, that means the public (small bets) loves Team A but the larger bettors don't agree. That gap (divergence D = money% - bets%) is the foundation of DK-level analysis.

**Philosophy:** DK data is RETAIL. It tells you where the public is, not where the market is going. DK confirms edge found elsewhere; it never discovers edge on its own.

---

### 1B. Pinnacle / Sharp Books (Layer 1 — Sharp Signal)

**What:** Lines and odds from the world's sharpest bookmakers
**Source:** OddsPapi API (primary), The-Odds-API Pinnacle (fallback)
**Frequency:** 3x daily (11:30 AM, 3:30 PM, 6:30 PM ET)
**File:** `data/l1_sharp.csv` + `data/l1_open_registry.csv`
**Budget:** 250 req/month (OddsPapi)

**Books available:** Pinnacle, Bookmaker.eu, Singbet, SBOBet, BetCris, Circa Sports
**Currently returning:** Pinnacle + Bookmaker.eu (2 of 6)

**What makes Pinnacle special:** Pinnacle accepts the largest bets in the world from the sharpest bettors. Their lines are considered the "true" market. When Pinnacle moves a line, it's because someone with real information bet enough to move it. This is the polar opposite of DK, where line moves might just be parlay whales.

**Key L1 features extracted:**
- **Move direction:** Did Pinnacle move toward or against this side?
- **Move magnitude:** How much did they move? (normalized 0-1 by sport/market)
- **Agreement:** Do multiple sharp books agree? (1.0x to 1.5x multiplier)
- **Speed:** Fast snap (sudden move) or slow grind? Speed signals information type.
- **Stability:** Are sharps holding their position or fluctuating?
- **Key number crossing:** Did the line cross 3, 7, 10, 14, or 17? (football only)
- **Leader detection:** Which book moved first? (Pinnacle leading = strongest)

---

### 1C. 31-Book Consensus (Layer 2 — Market Validation)

**What:** Lines and odds from every major US and European sportsbook
**Source:** The-Odds-API
**Frequency:** Synchronized with L1 (3x daily)
**File:** `data/l2_consensus.csv` + `data/l2_consensus_agg.csv`
**Budget:** 500 req/month free tier, 30-request reserve guard
**Books:** DraftKings, FanDuel, BetMGM, Bovada, William Hill, Unibet, Pinnacle, Betway, PointsBet, and 22 more

**What L2 provides:**
- **Consensus line:** Median line across all books (the "true" market number)
- **Dispersion:** How much do books disagree? Tight = confident market, wide = uncertain
- **Agreement with L1:** What % of books moved in the same direction as Pinnacle?
- **Pinnacle vs consensus:** Is Pinnacle ahead of or behind the market?
- **Stale price detection:** Is DK's line significantly behind the consensus?

**Why consensus matters:** If Pinnacle moves but no other book follows, maybe Pinnacle got bad information. If Pinnacle moves AND 20+ other books follow, the information is confirmed. L2 is the validation layer — it either amplifies or suppresses L1's signal.

---

### 1D. ESPN (Situational Context)

**What:** Free public data that provides game context
**Source:** ESPN public scoreboard API (no key required)
**Cache:** 30 minutes on disk

**Data pulled:**
| Data | Sport | How It's Used |
|------|-------|---------------|
| Injuries | NBA, NHL, NFL | Dampens confidence when key players are out |
| Rest days / B2B | NBA, NHL | Back-to-back penalty (-1.0) |
| Probable pitchers | MLB | ERA-based quality scoring, handedness |
| Probable goalies | NHL | Confirmed starter vs backup detection |
| AP/NET rankings | NCAAB | Ranking differential scoring |
| Final scores | All | Post-game grading for results_resolved.csv |

---

### 1E. Open-Meteo (Weather)

**What:** Free weather forecasting API
**Source:** `api.open-meteo.com` (completely free, no API key)
**Cache:** 30 minutes on disk
**Applies to:** NFL, NCAAF, MLB (outdoor stadiums only)

**Stadium database:** 62 venues (32 NFL + 30 MLB) with latitude, longitude, and dome flag stored in `weather.py`. Dome stadiums automatically return 0 adjustment.

**Weather signals:**
| Condition | Adjustment | Why It Matters |
|-----------|-----------|----------------|
| Wind >= 20 mph | -1.5 | Affects passing, kicking, fly balls |
| Wind 15-19 mph | -0.5 | Mild impact on game |
| Rain likely (>70%) | -1.0 | Affects ball handling, footing |
| Rain possible (40-69%) | -0.5 | Uncertainty |
| Extreme cold (<=20F) | -1.0 | Affects everything (NFL/NCAAF only) |
| Cold (21-32F) | -0.5 | Affects kicking, grip |

---

## PART 2: THE MERGE

**File:** `merge_layers.py`

Before scoring can happen, all data sources must be joined together into a single row per side. This is the merge step.

### How Games Are Matched Across Sources

The #1 challenge: DK calls a team "OKC Thunder -4.5", Pinnacle calls them "Oklahoma City Thunder", and ESPN calls them "Thunder". These are all the same team.

**Solution:** Canonical matching via `canonical_match.py` + `team_aliases.py`

```
Canonical Key = "{away_norm} @ {home_norm}|{sport}|{game_date}"
Example:       "atlanta hawks @ washington wizards|nba|2026-03-05"
```

**Team name normalization pipeline:**
1. Lowercase, strip special characters
2. Strip college mascots (NCAAB: "butler bulldogs" -> "butler")
3. Apply alias map (61 NBA/NHL + 100+ college aliases)
4. If exact match fails, fuzzy matching (>=0.8 score threshold)

### What Gets Joined

For each DK row, the merge attaches:
1. **L1 features** — sharp book signals (if available)
2. **L2 features** — consensus validation (if available)
3. **Layer mode** — L123/L13/L23/L3_ONLY based on what's available
4. **Injuries** — home/away injury lists and counts
5. **B2B flags** — HOME_B2B / AWAY_B2B / BOTH_B2B
6. **Weather** — wind, temp, precip for outdoor sports
7. **Sport context** — pitcher stats, goalie status, rankings

### Layer Mode

| L1 (Sharp) | L2 (Consensus) | Mode | Score Cap | Dashboard |
|------------|----------------|------|-----------|-----------|
| Yes | Yes | L123 | 100 | FULL |
| Yes | No | L13 | 85 | PARTIAL |
| No | Yes | L23 | 80 | PARTIAL |
| No | No | L3_ONLY | 75 | LIMITED |

More data = higher ceiling. You cannot get a STRONG_BET without all three layers.

---

## PART 3: SCORING

### The DK Base Score (dk_scoring.py)

Every row starts with a DK base score built from 16 components. This is the foundation — even L3_ONLY rows get this.

**Components (in order):**

| # | Component | Range | What It Does |
|---|-----------|-------|-------------|
| 1 | Dynamic base | 46-52 | Low-info early = 46, movement = 52, normal = 50 |
| 2 | Market read | -5 to +4 | Book's LINE RESPONSE to money (not the money itself) |
| 3 | Reverse Line Movement | 0 to +10 | Public heavy one way but book moves opposite |
| 4 | Regime classifier | sets mult | A/B/C/D/N — how much to trust divergence |
| 5 | Divergence (5-factor) | -5 to +12 | bets% vs money% gap, scaled by regime + intensity |
| 6 | Line movement | 0 to +8 | How much did DK's line move from open? |
| 7 | Key number crossing | 0 to +6 | Crossed 3 or 7 in football spread (massive signal) |
| 8 | Timing | 0 to +1 | MID = +1, others neutral |
| 9 | NCAAF early dampener | 0 to -2 | Early NCAAF splits unreliable |
| 10 | NCAAB single-market | 0 to -3 | Only one market available = weak signal |
| 11 | NHL puck line governor | 0 to -3 | All +-1.5 lines identical, prevents batch BET |
| 12a | Color classification | -7 to +9 | Concentrated money pattern, GATED by book response |
| 12b | ML vs Spread cross-check | -3 to +4 | Implied probability consistency |
| 12c | Line trajectory | -2 to +2.5 | MOVE_AND_HOLD / FLAT / SNAP_BACK / VOLATILE |
| 13 | ML risk governor | 0 to -6 | Heavy favorites/dogs get dampened |
| 14 | Longshot penalty | 0 to -10 | Sport-relative implied probability floor |
| 15 | ML-only penalty | 0 to -3 | ML moved but spread didn't |
| 16 | Retail alignment | 0 to -5 | bets >70% AND money >70% = likely wrong |

**The "Book Response" philosophy (the most important concept):**

DK money is recreational whales. What the BOOK does with that money tells you everything:
- Book CONFIRMS money (moves with it) = positive divergence (trust the signal)
- Book HOLDS (doesn't move) = mild negative (book absorbs, disagrees)
- Book FADES money (moves against it) = negative divergence (book says money is wrong)

This is why "Stealth Move" (concentrated money + book confirms) gets +4, while "Reverse Pressure" (concentrated money + book fades) gets -4. Same money, opposite book response, opposite score impact.

---

### The 11 Scoring Adjustments (scoring_v2.py)

On top of the DK base score, the unified scorer adds up to 10 adjustments:

```
raw_score = dk_base
  + l1_adj           # Sharp book signal (-5 to +10)
  + l2_adj           # Consensus validation (-5 to +7)
  + pattern_bonus    # Interaction pattern effect (-8 to +5)
  + cross_adj        # Spread/Total consistency (-2 to +1)
  + line_diff        # DK vs consensus gap (currently disabled)
  + decay            # Momentum decay (0 to -3)
  + b2b_adj          # Back-to-back fatigue (0 to -1)
  + injury_adj       # ESPN injury dampening (-2 to +1)
  + weather_adj      # Outdoor weather impact (-3.5 to 0)
  + sport_context_adj # MLB/NHL/NCAAB context (-3 to +3)
```

**L1 Adjustment (-5 to +10):**
The primary signal. Bidirectional — if sharp books moved AGAINST this side, L1 becomes a penalty. Components: magnitude, agreement, stability, speed, key numbers, leader book, DK money cross-check.

**L2 Adjustment (-5 to +7):**
Validates L1. If 60%+ of 31 books agree with Pinnacle's direction, L2 amplifies. If <30% agree, L2 suppresses. Dispersion trend (tightening/widening) adds +-1.5.

**Pattern Bonus (-8 to +5):**
Based on which interaction pattern is detected (see Part 4).

**Injury Adjustment (-2 to +1):**
NBA/NHL/NFL only. Our team injured = dampen. Opponent injured = slight boost. Totals: heavy combined injuries = dampen.

**Weather Adjustment (-3.5 to 0):**
NFL/NCAAF/MLB outdoor games. Wind, rain, extreme cold all reduce confidence. Cumulative.

**Sport Context (-3 to +3):**
- MLB: Pitcher quality differential (our ace vs their weak starter = +1.5) + park factors (Coors = over lean, Petco = under lean)
- NHL: Goalie confirmation (starter confirmed = +1.0, backup confirmed = -2.0)
- NCAAB: Ranking differential (#5 vs #22 = +1.0)

---

## PART 4: INTERACTION PATTERNS

Every scored row gets exactly ONE pattern (A through G, or N). This is the engine's classification of WHAT TYPE of edge exists. Patterns control score floors, caps, and STRONG_BET eligibility.

**Detection order:** F -> G -> A -> D -> B -> E -> C -> N (first match wins)

### Pattern A: SHARP_VS_PUBLIC (The Best Edge)
**Badge:** SHRP (green)
**What:** Pinnacle moved, consensus confirms (60%+ agreement), and DK public is betting the OTHER way.
**Why it's the best:** Sharp money is on one side, dumb money is on the other, and the broader market confirms the sharps. This is textbook edge.
**Score:** +5 bonus, floor 50, STRONG eligible

### Pattern G: REVERSE LINE MOVEMENT (The Sharpest Signal)
**Badge:** RLM (purple)
**What:** Public bets are 60%+ on one side, but the gap between bets% and money% is 15%+, sharps moved, and consensus confirms.
**Why it matters:** The book is literally moving the line AGAINST the side that 60%+ of tickets are on. They're accepting lopsided liability because they believe the other side is correct. This is the book telling you where to bet.
**Score:** +4 bonus, floor 50, STRONG eligible
**Strength formula:** `gap_factor x sharp_strength x consensus_agreement x bets_money_intensity`

### Pattern D: STALE_PRICE (DK Hasn't Caught Up)
**Badge:** STALE (red)
**What:** Sharp moved, consensus confirms, AND DK's line is significantly behind the consensus median.
**Why it matters:** DK updates slower than sharp books. If Pinnacle moved to -4.5 and consensus is at -4.0 but DK is still at -3.5, you're getting a number that the market has already moved past.
**Score:** +4 bonus, floor 50, STRONG eligible

### Pattern B: RETAIL_ALIGNMENT (Priced In)
**Badge:** ALGN (blue)
**What:** Sharp moved, consensus confirms, but public is also on the same side.
**Why it's capped:** Everyone agrees — which means the line has likely already adjusted. The edge is probably priced in.
**Score:** cap 70 (never STRONG), floor 45

### Pattern E: CONSENSUS_REJECTS (Market Doesn't Agree)
**Badge:** REJ (orange)
**What:** Sharp moved but less than 30% of the broader market followed.
**Why it's penalized:** Pinnacle moved but nobody else agreed. Maybe Pinnacle got bad info, or the move was noise. Significantly lower conviction.
**Score:** -6 penalty, cap 65, floor 40

### Pattern F: LATE_SNAP (Dangerous)
**Badge:** SNAP (amber)
**What:** Sharp books made a fast snap move with less than 1 hour to game.
**Why it's dangerous:** Late sharp moves can be information (injury news, lineup change) OR manipulation. High uncertainty. Never STRONG.
**Score:** -8 penalty, floor 40

### Pattern C: RETAIL_ALIGNMENT (Public Only)
**Badge:** PUB (gold)
**What:** No sharp movement detected, but DK public is heavily on one side.
**Why it's penalized:** Without sharp confirmation, heavy public action is usually wrong. This is where the retail bias trap lives.
**Score:** -5 penalty, floor 40

### Pattern N: NEUTRAL
**Badge:** NEU (gray)
**What:** No strong interaction pattern detected from any layer.
**Score:** 0 bonus, floor 40

---

## PART 5: DECISIONS

After scoring, every game-level row gets a decision label.

### Decision Thresholds

| Decision | Score | Net Edge (SPREAD/ML) | Net Edge (TOTAL) |
|----------|-------|---------------------|-------------------|
| **STRONG_BET** | >= 70 | >= 10 | >= 12 |
| **BET** | >= 67 | >= 10 | >= 12 |
| **LEAN** | >= 60 | any | any |
| **NO BET** | < 60 | any | any |

**Net edge** = difference between the highest and lowest side scores for a game/market. High net edge means one side is clearly favored over the other.

### STRONG_BET — The Highest Conviction

STRONG_BET is the engine's highest conviction play. All of these gates must pass:

1. **Layer mode = L123** — All three data layers present
2. **Pattern = A, D, or G** — Only the strongest signal patterns
3. **Score >= 70** — High absolute confidence
4. **Net edge >= 10/12** — Clear separation between sides
5. **Timing != LATE** — Not in the volatile final window
6. **Market read != "Public Drift"** — Not riding public momentum
7. **No cross-market contradiction** — Spread and total agree
8. **Persistence: strong_streak >= 2** — Must qualify for 2+ consecutive snapshots (NCAAB: 3)
9. **Stability: last_score near peak** — Score hasn't been declining
10. **No early block** — NCAAB/NCAAF can't STRONG in EARLY window

**Rarity target:** 5-10% of BET rows should be STRONG_BET.

### The Freeze Ledger

When a decision is made, it's frozen permanently in `decision_freeze_ledger.csv`.

**Rules:**
- Append-only (never delete rows)
- Dedup prefers higher decisions: STRONG_BET > BET > LEAN > NO BET
- Once a row reaches STRONG_BET, it NEVER downgrades
- Frozen fields: game_confidence, net_edge, game_decision, favored_side, decision_line, decision_odds
- This is the source of truth for grading and CLV

---

## PART 6: POST-GAME GRADING & CLV

### Results Resolution

After a game finishes:
1. ESPN final scores are pulled automatically
2. Each frozen decision is graded: WIN / LOSS / PUSH
3. CLV is calculated (did we beat the closing line?)
4. Results written to `results_resolved.csv`

### Closing Line Value (CLV)

CLV answers: "Did the engine consistently get better numbers than the market?"

**Decision line:** The DK line when the engine first made a BET/STRONG_BET decision. Captured in the freeze ledger.

**Closing line:** The last DK line before game start. Captured from the 10-minute scrape cadence.

**CLV Calculation:**

| Market | Formula | Example |
|--------|---------|---------|
| SPREAD | `decision - closing` | Got -3 when close was -3.5 = CLV +0.5 |
| TOTAL OVER | `closing - decision` | Took O 220 when close was 222 = CLV +2.0 |
| TOTAL UNDER | `decision - closing` | Took U 222 when close was 220 = CLV +2.0 |
| MONEYLINE | `closing_prob - decision_prob` | Got +150 when close was +130 = positive CLV |

**Why CLV matters more than win rate:** You can win 55% of bets and still lose money if you're getting bad numbers. Conversely, sustained positive CLV = sustained edge, even through losing streaks. CLV is the professional standard for evaluating betting models.

### KPIs (kpi_builder.py)

KPIs are read-only analytics over `results_resolved.csv`:

| KPI Dimension | Buckets |
|--------------|---------|
| Decision | STRONG_BET / BET / LEAN / NO BET |
| Confidence | 75+ / 70-74 / 65-69 / 60-64 / <60 |
| Net Edge | 13+ / 9-12 / 5-8 / 0-4 |
| CLV | >=2.0 / 1.0-1.9 / 0.1-0.9 / 0 / negative |
| Sport | NBA / NHL / MLB / NFL / NCAAB / NCAAF / UFC |
| Market | SPREAD / TOTAL / MONEYLINE |
| Pattern | A / B / C / D / E / F / G / N |

**KPI epoch:** Reset to 2026-03-05 (v2.0 baseline). Only counts the picked side.

---

## PART 7: SPORT-SPECIFIC INTELLIGENCE

### MLB

**Starting Pitcher Quality:**
- ERA mapped to -3.0 to +3.0 score
- Sample gate: W+L must be >= 3, otherwise SP_UNKNOWN (-1.0 penalty)
- For ML/SPREAD: pitcher quality differential (our pitcher - opponent) x 0.5
- For TOTAL: average quality of both pitchers x 0.3

**Park Factors (all 30 teams):**
- Coors Field: 1.28x (extreme hitter park — overs lean)
- Fenway Park: 1.10x (Green Monster — hitter friendly)
- Petco Park: 0.91x (pitcher park — unders lean)
- Only affects TOTAL market confidence

**Dashboard:** Pitcher badges show last name + ERA + handedness, colored green (ace) / gray (avg) / red (weak)

### NHL

**Goalie Confirmation:**
- Starter confirmed: +1.0 (line set for this goalie, trust it)
- Backup confirmed: -2.0 (line may not have adjusted, book value changes)
- Starter/backup inferred from W-L-OT record (20+ games = starter)
- Side-relative: opponent's backup = slight edge for us (+50% of penalty as bonus)

**Dashboard:** Goalie badges show name + status (confirmed = green, probable = yellow, unknown = gray)

### NCAAB

**Rankings:**
- AP/NET top 25 pulled from ESPN
- Rank gap >= 15: +/- 1.0 based on which side we're on
- Rank gap 8-14: +/- 0.5
- Ranked vs unranked: +/- 0.5

**Dashboard:** Purple rank badges (#5 vs #22)

### NFL / NCAAF

**Key number bonus:** +6 for spread crossing 3 or 7 (these are the most common margins of victory)
**Weather:** Full Open-Meteo integration for all outdoor stadiums
**Key numbers are football-only** — NBA/NHL/MLB do not have meaningful key numbers in spreads

### All Sports

**Injuries:** NBA/NHL/NFL get injury dampening (3+ injuries on our team = -2.0)
**B2B fatigue:** NBA/NHL back-to-back = -1.0

---

## PART 8: THE DASHBOARD

### What You See

The dashboard (`dashboard.html`) renders a table of every active game, sorted by score.

**Columns:**
Rank | Game + Badges | Market | Bets/Money | Open | Current | Edge | Confidence | Pattern | Layer | Timing | Decision

**Badge types on each game row:**
| Badge | When | Example |
|-------|------|---------|
| B2B | Team played yesterday | `HOME_B2B` (red) |
| Injuries | ESPN reports injuries | `H:3inj A:1inj` (red) |
| Weather | Outdoor + conditions | Wind icon / rain icon / sun icon |
| Dome | Indoor stadium | Stadium icon (gray) |
| Pitcher | MLB game | `RHP Cole 3.21` (green) / `LHP Smith 5.89` (red) |
| Park Factor | Extreme park | `1.28x` (gold) / `0.91x` (blue) |
| Goalie | NHL game | `Shesterkin` (green confirmed) |
| Ranking | NCAAB ranked teams | `#5 vs #22` (purple) |

### Pattern Badges
| Pattern | Badge | Color | Meaning |
|---------|-------|-------|---------|
| A | SHRP | Green | Sharp books vs public — best edge |
| B | ALGN | Blue | Everyone agrees — priced in |
| C | PUB | Gold | Public only — likely wrong |
| D | STALE | Red | DK line is behind — exploitable |
| E | REJ | Orange | Consensus rejects sharp — low conviction |
| F | SNAP | Amber | Late sharp snap — dangerous |
| G | RLM | Purple | Reverse line movement — strongest signal |
| N | NEU | Gray | No pattern — neutral |

### Layer Badges
| Mode | Badge | Color | Meaning |
|------|-------|-------|---------|
| L123 | FULL | Green | All 3 data layers — highest ceiling (cap 100) |
| L13 | PARTIAL | Amber | Sharp + DK only (cap 85) |
| L23 | PARTIAL | Blue | Consensus + DK only (cap 80) |
| L3_ONLY | LIMITED | Gray | DK data only (cap 75) |

---

## PART 9: ALL FILES

### Data Files (data/ directory)

| File | What It Contains | Written By | Frequency |
|------|-----------------|-----------|-----------|
| `snapshots.csv` | Every DK scrape — bets%, money%, lines, odds | DK scraper | Every 10 min |
| `l1_sharp.csv` | Pinnacle + sharp book lines (append-only) | l1_scraper.py | 3x daily |
| `l1_open_registry.csv` | First-seen lines for computing moves | l1_scraper.py | 3x daily |
| `l2_consensus.csv` | Raw 31-book lines (append-only) | l2_scraper.py | 3x daily |
| `l2_consensus_agg.csv` | Aggregated consensus (median, std, n_books) | l2_scraper.py | 3x daily |
| `dashboard.csv` | Game-level summaries for UI | aggregation | Each run |
| `decision_snapshots.csv` | Ephemeral side-level evaluations | build_dashboard | Each run |
| `decision_freeze_ledger.csv` | Permanent decision record (never downgrade) | freeze step | Each run |
| `results_resolved.csv` | Graded results + CLV + frozen decisions | outcomes | Each run |
| `final_scores_history.csv` | ESPN final scores | outcomes | Each run |
| `row_state.csv` | Persistent side-level memory (peak, streak) | metrics tap | Each run |
| `signal_ledger.csv` | Event log of threshold crossings | metrics tap | Each run |
| `espn_cache/*.json` | Cached ESPN API responses | espn_situational | 30-min TTL |
| `weather_cache/*.json` | Cached Open-Meteo responses | weather.py | 30-min TTL |

### Code Modules

| File | Lines | What It Does |
|------|-------|-------------|
| `main.py` | ~5900 | Core pipeline: ingest -> score -> aggregate -> freeze -> dashboard -> outcomes. The orchestrator. |
| `scoring_v2.py` | ~670 | Unified scoring: 11 adjustment components + pattern detection + floors/caps |
| `dk_scoring.py` | ~500 | DK base score: 16 components including book response, regime, trajectory |
| `l1_features.py` | ~300 | Sharp book feature extraction: direction, magnitude, agreement, speed, stability |
| `l2_features.py` | ~250 | Consensus features: dispersion, agreement, trend, Pinnacle proximity |
| `dk_rules.py` | ~200 | DK retail rules: divergence threshold, timing credibility, alignment penalties |
| `merge_layers.py` | ~400 | The big join: L1 + L2 + ESPN + weather + sport context onto DK rows |
| `engine_config.py` | ~250 | Every magic number, threshold, file path, API config, season calendar |
| `team_aliases.py` | ~400 | Team name normalization. 200+ aliases across all sports + college mascot stripping |
| `canonical_match.py` | ~200 | Cross-source game matching. Canonical key builder + fuzzy fallback |
| `espn_situational.py` | ~400 | ESPN free API: injuries, rest days, pitchers, goalies, rankings |
| `weather.py` | ~220 | Open-Meteo weather + 62 stadium database |
| `mlb_context.py` | ~190 | SP quality scoring (ERA-based) + park factors (30 teams) |
| `nhl_context.py` | ~130 | Goalie confirmation scoring + starter/backup detection |
| `kpi_builder.py` | ~300 | Read-only KPI analytics + CLV analysis |
| `odds_api.py` | ~200 | The-Odds-API wrapper (budget guard, caching) |
| `l1_scraper.py` | ~300 | OddsPapi wrapper + The-Odds-API fallback for L1 |
| `l2_scraper.py` | ~250 | Consensus aggregation from The-Odds-API |
| `serve.py` | ~100 | HTTP server for live dashboard |

---

## PART 10: THE EVOLUTION (v1.0 to v2.1)

### v1.0 — The Beginning (Feb 27, 2026)
**Architecture:** Single-layer. DK data only.
**Scoring:** Fixed base 50 + additive signals. Color classification (DARK_GREEN/LIGHT_GREEN/RED). Simple divergence always added to score.
**Problem:** Everything that looked like divergence got scored the same, whether the book confirmed it or not.

### v1.1 — STRONG_BET Wiring (Mar 1, 2026)
**Added:** STRONG_BET decision label with eligibility gates. Canonical key system for consistent row identity. Persistence cap (+6/tick) to prevent score explosions. SPREAD dampening (x0.92). ML risk governor.
**Problem:** Still single-layer (DK only). Divergence was always trusted regardless of book behavior.

### v1.2 — Regime-First Scoring (Mar 2, 2026)
**Added:** Regime classifier (A/B/C/D/N) — finally asking "what is the BOOK doing?" before trusting divergence. Combined 5-factor divergence multiplier. Dynamic base (44/50/52). RLM as first-class signal. Contradiction penalty (-4 instead of 0).
**Problem:** Still single-layer. All signals came from DK, which is retail. No independent confirmation.

### v2.0 — Three Layers (Mar 4, 2026)
**The big rewrite.** Three independent data sources:
- L1: Pinnacle sharp books (the signal)
- L2: 31-book consensus (the validation)
- L3: DK retail behavior (the modifier)

**Added:** Pattern detection (A-G). Layer mode caps. Cross-layer interaction scoring. DK demoted from signal to modifier (-10 to +10). STRONG requires L123 + Pattern A/D/G. New modules: scoring_v2.py, dk_scoring.py, l1_features.py, l2_features.py, merge_layers.py.

**Philosophy shift:** "Read the book, not the bettors." DK money is retail. The book's line response to that money is what matters.

### v2.0.1 — Scoring Signal Overhaul (Mar 5, 2026)
**Refined:** Book response philosophy fully wired into dk_scoring.py. Continuous signals replacing binary thresholds. ML vs Spread implied probability cross-check. Line movement trajectory tracking (MOVE_AND_HOLD, FLAT, SNAP_BACK, VOLATILE).

### v2.1 — Situational Intelligence (Mar 5, 2026)
**Added:** CLV tracking (decision line + closing line + CLV calculation). ESPN injury counts wired into scoring. Weather integration via Open-Meteo (62 stadiums). MLB starting pitcher quality + park factors (30 teams). NHL goalie confirmation scoring. NCAAB ranking differential. Score formula now has 11 adjustment components. Dashboard badges for everything.

---

## PART 11: KEY PRINCIPLES

### The 11 Engine Laws

1. **L1 leads.** Sharp books provide the primary signal. DK confirms, never discovers.
2. **L2 validates.** Consensus either confirms or rejects L1. This determines confidence.
3. **L3 modifies.** DK retail behavior adjusts score within bounds, never overrides L1/L2.
4. **Patterns classify.** Every scored row gets exactly one pattern (A-G or N).
5. **Layer mode caps.** More data = higher ceiling. L3-only rows cannot exceed 75.
6. **STRONG requires FULL.** STRONG_BET needs all 3 layers + Pattern A/D/G.
7. **Scoring happens once.** No downstream layer recomputes scores.
8. **Freeze is permanent.** Decisions freeze at dashboard build time; never downgraded.
9. **Outcomes grade only.** Post-game results do not influence scoring or decisions.
10. **KPIs analyze only.** Read-only layer. Counts only the picked side.
11. **Context adjusts, never overrides.** Weather, injuries, and sport context modify within bounds.

### The Single Writer Rule

Every data file has exactly one writer. No exceptions.

| File | Writer |
|------|--------|
| snapshots.csv | DK scraper |
| l1_sharp.csv | l1_scraper.py |
| l2_consensus*.csv | l2_scraper.py |
| dashboard.csv | aggregation step |
| row_state.csv | metrics tap |
| signal_ledger.csv | metrics tap |
| final_scores_history.csv | outcomes |
| results_resolved.csv | outcomes |
| decision_freeze_ledger.csv | freeze step |

### The Canonical Grain

Every analytical operation — scoring, metrics, grading, KPIs — operates on the same grain:

**(sport, game_id, market, side)**

Game-level views are derived. Wide views are UI-only. No other grain is canonical.

---

## PART 12: INFRASTRUCTURE

### Server
- **IP:** 159.65.167.146
- **SSH Port:** 2222 (not default 22)
- **Path:** /opt/red-fox-market-dynamics
- **Python:** .venv/bin/python
- **Auto-deploy:** `git pull origin main` every 1 minute via cron

### Cron Schedule (all times ET)

| What | When | Command |
|------|------|---------|
| DK scraper | Every 10 min | `run_all_sports.sh` |
| L1 + L2 odds pull | 11:30 AM | `odds_snapshot_all` |
| L1 + L2 odds pull | 3:30 PM | `odds_snapshot_all` |
| L1 + L2 odds pull | 6:30 PM | `odds_snapshot_all` |
| NFL Sunday extra | 12:15 PM (Sep-Feb) | `odds_snapshot_all` |
| Auto-deploy | Every 1 min | `git pull origin main` |

### API Budgets

| API | Monthly Limit | Reserve | Cost |
|-----|--------------|---------|------|
| The-Odds-API (L2) | 500 requests | 30 reserve | Free |
| OddsPapi (L1) | 250 requests | 20 reserve | Free |
| ESPN | Unlimited | — | Free |
| Open-Meteo | Unlimited | — | Free |
| DraftKings scrape | Unlimited | — | Free |

### Sport Season Calendar

| Sport | Active Months | L1 Tournament ID |
|-------|--------------|------------------|
| NBA | Oct - Jun | 132 |
| NHL | Oct - Jun | TBD |
| MLB | Mar - Nov | TBD |
| NFL | Sep - Feb | TBD |
| NCAAB | Nov - Apr | 648 |
| NCAAF | Aug - Jan | TBD |
| UFC | Year-round | TBD |

---

## PART 13: WHAT'S NEXT

### Phase 7 — Statistical Validation (Needs 100+ Graded Decisions)

| Feature | What It Does | Why Wait |
|---------|-------------|----------|
| Score Calibration | Map raw 0-100 scores to actual win probabilities | Need enough data points to build reliable curves |
| Kelly Criterion | Optimal bet sizing based on calibrated edge | Requires calibration first |
| Bayesian Updates | Prior from historical patterns, posterior from live signals | Need historical distribution |

These are deferred until the engine has accumulated enough resolved BET and STRONG_BET decisions with known outcomes to build statistically meaningful calibration curves. Estimated timeline: 2-4 weeks of daily operation.

---

*This document covers the complete Red Fox system as of v2.1 (2026-03-05). For version-specific architecture details, see the individual spec files: Engine Arch 1.1.txt, Engine Arch 1.2.md, and Engine Arch 2.0.md (updated to v2.1).*
