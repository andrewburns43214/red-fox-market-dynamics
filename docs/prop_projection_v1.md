# Red Fox Market Intelligence — Props-Only Projection v1

## Isolation contract

The subsystem reads PropLine player props, official roster sources, MLB game
context, and final scores. It does not read `snapshots.csv`, spread, moneyline,
game total, team total, public splits, Market Read, Supported Side, Market Rank,
or Red Fox Favorite. Its only public artifact is `data/prop_projections.json`.
Failure, timeout, missing credentials, or inadequate coverage cannot block the
normal Red Fox refresh. Customer-facing states are `Props-Only · High`,
`Props-Only · Moderate`, `Props not open yet`, and `Insufficient coverage`.

The API key is read only from `PROPLINE_API_KEY` and is sent only in the
`X-API-Key` request header. It is never serialized, logged, published, or
included in a URL.

## Versioned model pipeline

1. Reject wrong-event players, players not on an official current roster,
   ambiguous identities, suspended legs, stale legs, malformed prices, and
   one-sided Over/Under markets.
2. Pair Over and Under at `(event, book, player, stat, threshold)` and remove
   the book hold: `p_over_fair = p_over / (p_over + p_under)`.
3. Choose one canonical threshold per player/stat: most traditional books,
   then closest to the center of observed thresholds, then lower threshold as
   a deterministic tie-break. Alternate and milestone markets are excluded.
4. Convert the canonical threshold to an expected player stat with
   `mean = threshold + sigma_stat * NormalInverse(p_over_fair)`. Fixed sigma
   values live in `prop_projection_config.py` and change only with a new model
   version.
5. Aggregate player production once per statistical family. Correlated markets
   validate or form a blended estimator; they are never independently summed.
6. Convert team production to a team-score mean, then round only for display.
   The unrounded means and version remain in the audit ledger.

### NFL / NCAAF: `prop_projection_nfl_v1`, `prop_projection_ncaaf_v1`

- Passing TD expectation is the primary passing scoring component.
- Rushing TD props are summed once. If absent, verified rushing-yard means use
  the explicit fallback `rush_yards / 92`.
- Receiving TDs validate passing TDs and are not added again. NFL anytime,
  first, and 2+ scorer markets are collected in the private event cache for
  research but do not change v1 scores: most are one-sided and cannot use the
  paired Over/Under de-vig formula. The rushing-yards / 92 component is
  disclosed as a proxy in public scoring coverage.
- Kicking points are used directly. If absent, `3 * field_goals + PATs` is the
  non-overlapping fallback.
- Team mean: `6 * (passing_TD + rushing_TD) + kicking_points`, bounded to
  6–45. Yardage, receptions, attempts, and targets drive coverage and
  validation but do not duplicate direct scoring props.
- Score standard deviation: 6.8 points per team.

### MLB: `prop_projection_mlb_v1`

Four non-additive run estimators are constructed when their required inputs
exist:

- `1.12 * sum(batter runs)`
- `1.06 * sum(batter RBI)`
- `0.17*H + 0.13*TB + 0.42*HR + 0.22*BB`
- opposing starter earned runs plus `0.465` runs per projected bullpen inning,
  where bullpen innings are `9 - pitcher_outs/3`

The team mean is the median of available estimators, bounded to 1–10. This
prevents hits, total bases, home runs, runs, RBI, and pitcher earned runs from
being naively added together. Score standard deviation is 2.1 runs per team.

## Market policy

| Sport | Core | Supplemental | Correlated / validation | Low value / excluded |
|---|---|---|---|---|
| NFL/NCAAF | pass yards/TD/INT, rush yards, receiving yards/receptions, rush/receiving TD, kicking points | attempts, completions, carries, targets, FG, PAT | receiving TD vs pass TD; receiving yards vs receptions/targets | longest, first TD, 2+/3+ milestones, combo markets, defense props |
| MLB | pitcher outs/hits/walks/ER; hitter hits/TB/HR/runs/RBI | pitcher K, batter K/BB/SB | H/TB/HR and runs/RBI/pitcher ER blended estimators | H+R+RBI, singles/doubles/triples as independent totals, milestones |
| NBA/NCAAB (disabled) | points, rebounds, assists, 3PM, FG, FT | turnovers, steals, blocks | combo markets | alternates and milestones |
| NHL (disabled) | goalie saves, shots, goals, assists, points | power-play points, blocks | goal scorer markets | alternates and milestones |

## Coverage gate

Thresholds are intentionally conservative and may be tightened without
changing scoring coefficients.

- Football moderate: each team has at least four verified players, two books,
  three families, passing-TD coverage, rushing TD or yardage coverage, and
  direct kicker points or both field-goal and extra-point coverage. High additionally
  requires seven players, three books, four families, a passing-TD line from
  at least three books, kicking from at least two books, and two directly
  priced rushing-TD players with at least two books each per team. The oldest
  qualifying line must be no more than 15 minutes old for NFL or 30 minutes
  for NCAAF.
- MLB moderate: both probable pitchers are verified, each team has at least
  seven represented batters, two books, and three families. A game also earns
  Moderate when one probable pitcher is represented and the other is either
  unannounced or lacks props, provided both teams have nine represented hitters,
  two books, and both hitting and run-creation families. High requires both pitchers, a confirmed
  lineup, eight batters, three books, and four families.
- Anything below moderate is `INSUFFICIENT`; no projected score is published.
  NCAAF uses the same gate as NFL and never receives a weaker exception.

The broad configured market surface is always collected. Public explanations
show only five to seven balanced, strongest anchors. Correlated families may
validate or contribute to a blended estimator, but are never independently
summed as separate production.

## Collection and storage

- One bulk sport request is used when a sport is due.
- NFL more than 24 hours out: hourly. Within 24 hours: every 15 minutes.
  Final hour: every 10 minutes. Other enabled sports keep the prior schedule:
  more than 6 hours out hourly, six to one hours every 30 minutes, and final
  hour every 12 minutes. Collection stops at start. NFL line validity is
  capped at 25 minutes inside 24 hours and 15 minutes in the final hour.
- The local hard cap is 190 PropLine requests per UTC day. Provider quota
  headers are persisted after every request. During dense overlapping months,
  the expected event-aware range is about 80–150 requests/day, with the hard
  stop preventing a runaway loop.
- MLB context is cached for six hours, then 30 minutes in the final hour.
- Raw event observations are SHA-256 hashed. The latest bulk response is
  overwritten per sport; the append-only audit records only compact event
  metadata and the hash when it changes. Hash/context state expires after 14
  days, avoiding a duplicate raw-payload archive.
- Public projections are atomic JSON. Raw/canonical lines, projection ledger,
  quota state, and resolution performance remain under the private data tree.
- Football HIGH confidence requires fresh, well-sourced passing and kicking
  lines plus at least two directly priced rushing-TD players per team. A score
  using the rushing-yard proxy remains MODERATE even with broad player
  coverage. Source age uses the oldest contributing Over/Under leg across the
  selected book pairs and advances on the public card between refreshes. The
  card shows expected scores to two decimals and the rushing proxy when used.
- Final scores resolve projected home/away error, MAE, and bias privately every
  six hours when unresolved games exist.
