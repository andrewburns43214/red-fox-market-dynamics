# KPI Integrity Contract

This measurement layer does not change snapshot collection, market scoring,
board ranking, Red Fox Favorite qualification, or prop projection output.

## Official records

- Red Fox Favorite v2 and v3 share one continuous official W-L-P and ROI record.
- `favorite_rule_version` remains internal cohort metadata.
- Anomaly observations remain append-only, but performance KPIs use exactly one
  official action per sport, event ID, and market: the first KPI-eligible action
  ordered by signal time, capture time, then action ID.
- Opposite or later actions remain available for lifecycle research and cannot
  enter official performance totals.

## Grading evidence

- Missing, non-numeric, or non-finite scores always grade `UNRESOLVED`.
- Every graded action stores the final score, source, provider event ID when
  available, score-resolution time, grading time, and an evidence hash.
- Final-score history is append-only and first-write-wins. A routine snapshot
  refresh cannot silently delete or replace historical final-score evidence.
- Report outputs are written with atomic file replacement.

## Cohort interpretation

Favorite cohorts include sport, market, pathway, Market Read, bet and money
split bands, price/spread bands, movement magnitude, whipsaw state,
qualification timing, cross-market state, T-20 state, locked state, and rule
version.

Sample labels are descriptive safeguards, not automatic optimization rules:

- `TOO_SMALL`: fewer than 10 graded records
- `EARLY`: 10-24
- `DEVELOPING`: 25-49
- `MATURE`: 50 or more

No cohort result automatically changes Favorite thresholds. The required flow
is observe, investigate, replay/shadow test, and deliberately approve.

## Confidence and props

The general engine `game_confidence` field is treated as signal strength until
out-of-sample calibration establishes probability meaning. Prop projection v1
and v2 shadow evaluation remains separate from Favorite reporting.
