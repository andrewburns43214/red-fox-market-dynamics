# Red Fox Props-Only Projection v2 — Private Shadow

## Safety and isolation

`prop_projection_v2` is research-only. It receives the exact canonical lines
already validated by v1 and does not make an additional provider request. Its
records are written only to the private prop data directory. It does not enter
`data/prop_projections.json`, the customer board, Red Fox Favorite, Market Read,
or any ranking calculation. A shadow exception is caught and recorded without
interrupting v1 or the normal RF runner.

The public v1 model remains unchanged while v2 accumulates paired closing
projections and final scores.

`prop_projection_service.py shadow-backfill` converts the already-private v1
canonical ledger into v2 records without making an API call or writing the
public projection file. Resolution selects the chronologically latest pregame
shadow record even if older observations were backfilled later.

## Distribution contract

Every configured prop key is explicitly classified in
`prop_projection_v2_config.py`:

- `CONTINUOUS`: yardage statistics. The initial baseline retains configurable
  Normal sigma assumptions and records the conversion used.
- `COUNT`: integer counts. The baseline solves for the Poisson mean whose tail
  probability at the posted threshold equals the de-vigged Over probability.
  For line `L`, `k = floor(L) + 1`, and the solver enforces the universal bound
  `E[X] >= k * P(X >= k)`.
- `BINARY_EVENT`: yes/no event markets. The expected indicator equals the
  de-vigged event probability. Batter HR is intentionally treated as HR
  occurrence in the v2 offensive estimator, not as a continuous 0.5 mean.

Poisson is a defensible mathematical baseline, not a claim that every count is
truly Poisson. Negative-binomial and zero-inflated candidates can be added and
compared once the settled sample is large enough.

## NFL shadow decomposition

For each team, v2 records:

- passing-touchdown expectation from the direct passing-TD count market;
- direct rushing-TD props when present;
- a scorer-TD-surface alternative that combines rushing- and receiving-TD
  expectations instead of adding either to passing TDs;
- the existing `sum(rushing yards) / 92` candidate, explicitly labeled
  `unvalidated`;
- a league rushing-TD-rate candidate;
- direct kicker-points expectation;
- an FG-plus-XP reconstruction when both pieces exist;
- a historical kicking baseline when the prop surface is incomplete; and
- a league residual-scoring baseline.

Receiving TDs are never added to passing TDs. The residual baseline covers
non-pass/rush touchdowns, safeties, successful two-point tries, and defensive
conversion returns.

The league baselines come from 2021–2025 regular-season nflverse play-by-play,
2,718 team-games, reproduced by `tools/research_nfl_residual.py`:

- rushing TD: `0.9135393672` per team-game;
- kicking: `7.1688741722` points per team-game;
- residual scoring: `1.1111111111` points per team-game.

Source: [nflverse play-by-play releases](https://github.com/nflverse/nflverse-data/releases/tag/pbp), using the documented fields in the [nflreadr play-by-play dictionary](https://nflreadr.nflverse.com/articles/dictionary_pbp.html).

The shadow ledger retains a score variant for every rushing method available to
both teams plus a median consensus candidate. No estimator is promoted based on
the initial replay.

### Kicking confidence

Shadow High requires either:

- direct kicker-points coverage from at least two books; or
- both FG and XP props, each from at least two books.

FG-only, XP-only, or one-book scoring coverage selects the labeled historical
kicking baseline and caps shadow confidence at Moderate.

## MLB shadow decomposition

Discrete conversion is applied before constructing four separate estimates of
the same latent team-run expectation:

1. sum of batter expected runs;
2. converted batter RBI expectation;
3. a hits / total-bases / HR-event / walks offensive-production estimator; and
4. opposing starter expected ER plus an explicit bullpen-innings component.

These estimates are never added. The ledger records three competing robust
combinations and a consensus candidate:

- simple median;
- estimator-quality weighted median;
- reliability-weighted mean after MAD-based outlier clipping; and
- the median of those candidates (`consensus_candidate`).

Estimator quality reflects represented-player coverage and book depth. Probable
pitcher identity is used when available. Starter outs determine the innings not
covered by the starter; those innings receive a separately recorded bullpen
rate candidate. The inherited `0.465` runs per bullpen inning remains explicitly
versioned and pending outcome calibration.

## Outcome evaluation

The resolution pass uses the same final-score response already requested for
v1. It adds no provider request. For every v2 variant it stores and reports:

- team-score MAE and bias;
- total-score MAE and bias;
- margin MAE;
- decisive winner accuracy and sample count; and
- confidence and the paired v1 closing benchmark.

Artifacts remain private:

- `projection_v2_shadow_ledger.jsonl`
- `projection_v2_shadow_errors.jsonl`
- `resolution_v2_shadow_state.json`
- `performance_v2_shadow.json`

No v2 candidate is eligible for customer display until an explicit promotion
decision is made from out-of-sample results.
