# Red Fox Props-Only Projection v3 — Verified Consensus

## What publishes

V3 is the customer-facing model for NFL, NCAAF, and MLB. It treats coverage
and projection quality as separate gates: a game can have many fresh props and
still remain unavailable when independent score estimators are incomplete or
disagree. NBA, NCAAB, NHL, and UFC remain disabled until a sport-specific
model and provider acceptance sample are available.

The model uses player props only. It does not read the game spread, total,
moneyline, public splits, Market Read, rank, or Red Fox Favorite.

## Distribution conversion

Paired Over/Under prices are de-vigged at a common player/stat threshold.
Continuous yardage props retain their versioned Normal conversion. Integer
count props—including touchdowns, kicking, batter statistics, and pitcher
statistics—use Poisson tail inversion. Binary occurrence props use the
de-vigged event probability. This replaces the v1 practice of applying a
continuous Normal approximation to every statistic.

## NFL and NCAAF

Both teams must have two complete scoring reconstructions:

1. A paired-prop method using passing TDs, at least two directly priced
   rushing-TD players, and reliable kicking points (or paired FG plus XP).
2. An independent touchdown-scorer surface using at least five verified
   anytime-TD players, at least two books per included player, and at least
   three books across each team. Anytime, 2+, and 3+ rungs are converted to
   player touchdown means; first-TD prices are never used.

One-sided scorer markets cannot be exactly de-vigged without a No price. V3
therefore uses the best available price across books—the lowest raw implied
probability—as the conservative observable estimate. Ladder probabilities are
forced to be monotonic before Poisson conversion.

The two team estimates must be within 4.5 points, or the score is rejected as
`projection_methods_disagree`. The published mean is their average. V3 does
not use the old `rushing yards / 92` shortcut and does not fill a missing
component with a league scoring baseline.

## MLB

V3 builds four independent run estimators after discrete prop conversion:

- batter expected runs;
- converted batter RBI;
- a correlation-controlled hits/total-bases/home-run/walk estimator; and
- opposing starter earned runs plus projected bullpen innings.

At least three estimators must exist for each team. The tightest set of three
must fit within 2.1 runs (one published team-score standard deviation). This
allows one visibly recorded outlier but prevents a broad, internally
inconsistent prop surface from producing a score. The final value is the
existing robust consensus candidate; the independent estimates are never
summed.

## Publication and history

Qualified cards display `Props-Only · Verified High/Moderate` and expose only
the true scoring anchors. Rejected cards distinguish missing independent
methods from excessive disagreement. A temporary feed loss can retain the
last available V3 read, but a v1 score is never carried across the model
migration. Existing ledger rows and historical customer records are not
rewritten; all new records use `prop_projection_<sport>_v3_consensus_1`.
Each new private ledger row also keeps the corresponding v1 result as a
benchmark, while that benchmark is removed from the public JSON.

V3 is intentionally selective. `Verified` means the input, identity,
freshness, completeness, and cross-method agreement gates passed. It does not
claim a calibrated accuracy level; outcome calibration continues in the
private resolution ledger.
