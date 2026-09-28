# Red Fox Props-Only Projection v3 — Best Available

## What publishes

V3 is the customer-facing model for NFL, NCAAF, and MLB. It treats the score
and its quality as separate outputs. Whenever both teams have a usable
prop-derived scoring reconstruction, the best estimate is displayed. Complete
independent agreement earns a Verified label; thinner or conflicting evidence
is labeled Best Available instead of hiding the score. NBA, NCAAB, NHL, and
UFC remain disabled until a sport-specific scoring model exists.

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

V3 attempts two scoring reconstructions:

1. A paired-prop method using passing TDs, at least two directly priced
   rushing-TD players, and reliable kicking points (or paired FG plus XP).
2. An independent touchdown-scorer surface using at least five verified
   anytime-TD players. Anytime, 2+, and 3+ rungs are converted to
   player touchdown means; first-TD prices are never used.

One-sided scorer markets cannot be exactly de-vigged without a No price. V3
therefore uses the best available price across books—the lowest raw implied
probability—as the conservative observable estimate. Ladder probabilities are
forced to be monotonic before Poisson conversion.

The touchdown-scorer surface is the primary score because it prices scoring
directly. Paired passing/rushing/kicking props are an independent confirmation.
When both are complete, use direct rushing TDs, have at least three scorer
books, and agree within 4.5 points, the score is Verified. Otherwise the
strongest usable reconstruction is still shown as Best Available. The old
`rushing yards / 92` calculation is retained only as a disclosed last-resort
structural fallback; it never earns Verified status.

## MLB

V3 builds four independent run estimators after discrete prop conversion:

- batter expected runs;
- converted batter RBI;
- a correlation-controlled hits/total-bases/home-run/walk estimator; and
- opposing starter earned runs plus projected bullpen innings.

The final value is the robust consensus candidate; the independent estimates
are never summed. Three estimators fitting within 2.1 runs earn Verified.
Thinner or wider evidence still displays the best consensus with a Best
Available warning. No score is shown only when one team has no usable run
estimator.

## Publication and history

Cards display `Props-Only · Verified High/Moderate` or `Props-Only · Best
Available` and expose only the actual scoring anchors. A temporary feed loss
can retain the last available V3 read, but a v1 score is never carried across the model
migration. Existing ledger rows and historical customer records are not
rewritten; all new records use `prop_projection_<sport>_v3_best_1`.
Each new private ledger row also keeps the corresponding v1 result as a
benchmark, while that benchmark is removed from the public JSON.

`Verified` means the input, identity, freshness, completeness, and cross-method
agreement gates passed. `Best Available` means the displayed number is still
the system's preferred prop-only estimate, but confirmation is incomplete or
conflicting. Neither label claims a calibrated accuracy level; outcome
calibration continues in the private resolution ledger.
