# Pressure / Resistance Role Audit — 2026-09-06

## Scope and frozen production input

- Production revision before the change: `0a6d559`
- Snapshot captured: 2026-09-06 12:19 ET
- Population: 175 published markets / 350 paired sides
- Roles are presentation evidence generated after classification. They do not contribute to classification, scoring, board ordering, or directional-lean qualification.

## Existing behavior

`Pressure Side` was assigned only through the `Public Pressure` context chip. `Resistance Side` was then assigned to the counterpart only when the pair contained exactly one `Freeze`. This coupled the resistance role to a classification instead of the actual response.

Before counts:

| Role | Sides | Classification breakdown |
|---|---:|---|
| Pressure Side | 55 | Freeze 50; Follow 4; Watch 1 |
| Resistance Side | 50 | Watch 45; Contrarian 5 |

The four Follow markets correctly had Pressure Side without Resistance Side. Louisville/Ole Miss spread also had Pressure Side without Resistance Side solely because its pressured side was Watch rather than Freeze, despite an adverse response and a Contrarian counterpart.

## Generalized role rule

A side qualifies as `Pressure Side` when both bets and money are at least 55% and at least one is 70% or higher. Capped-split and Heavy Favorite suppression remains respected. This symmetric role-only threshold captures ticket-led and money-led concentration without creating a new classification.

The counterpart qualifies as `Resistance Side` only when the pressure-side evidence is adverse (`AGAINST`), held (`Held`), or already classified as Freeze. Follow/TOWARD pressure does not create resistance. A merely limited favorable response and an unresolved active Whipsaw do not create resistance.

## Production impact simulation

After counts on the frozen snapshot:

| Role | Before | After | Added |
|---|---:|---:|---:|
| Pressure Side | 55 | 76 | 21 |
| Resistance Side | 50 | 68 | 18 |

All 21 added Pressure Sides are Watch: 6 Spread, 11 Total, and 4 Moneyline. The 18 added Resistance Sides are 15 Watch and 3 Contrarian: 7 Spread, 8 Total, and 3 Moneyline.

| Market | Pressure Side added | Response | Resistance Side added |
|---|---|---|---|
| SF Giants @ NY Mets — Total | Over 8 (67% / 93%) | AGAINST | Under 8 (Contrarian) |
| WAS Nationals @ LA Dodgers — Total | Over 8 (61% / 90%) | TOWARD | — |
| NY Jets @ TEN Titans — Total | Under 39.5 (55% / 91%) | AGAINST | Over 39.5 (Contrarian) |
| TB Rays @ TEX Rangers — Moneyline | TB Rays (69% / 94%) | LIMITED / Held | TEX Rangers |
| ARI Diamondbacks @ HOU Astros — Total | Over 8 (60% / 71%) | LIMITED / Held | Under 8 |
| SF Giants @ NY Mets — Spread | NY Mets -1.5 (62% / 92%) | LIMITED / Held | SF Giants +1.5 |
| ATL Falcons @ PIT Steelers — Total | Under 42.5 (56% / 94%) | LIMITED / Whipsaw | — |
| Tommy McMillen vs Marwan Rahiki — Moneyline | Tommy McMillen (69% / 78%) | LIMITED favorable | — |
| Duke @ Illinois — Total | Over 54.5 (57% / 76%) | AGAINST / Held | Under 54.5 |
| DET Tigers @ CLE Guardians — Spread | CLE Guardians -1.5 (69% / 97%) | LIMITED / Held | DET Tigers +1.5 |
| WAS Nationals @ LA Dodgers — Spread | LA Dodgers -1.5 (69% / 86%) | LIMITED / Held | WAS Nationals +1.5 |
| STL Cardinals @ COL Rockies — Spread | STL Cardinals -1.5 (67% / 88%) | LIMITED / Held | COL Rockies +1.5 |
| MIN Twins @ CHI White Sox — Spread | CHI White Sox -1.5 (58% / 76%) | LIMITED / Held | MIN Twins +1.5 |
| NY Yankees @ SD Padres — Total | Over 7 (58% / 70%) | LIMITED / Held | Under 7 |
| STL Cardinals @ COL Rockies — Total | Over 11 (60% / 85%) | LIMITED / Held | Under 11 |
| Oklahoma @ Michigan — Moneyline | Oklahoma (67% / 97%) | LIMITED / Held | Michigan |
| WAS Commanders @ PHI Eagles — Total | Under 44.5 (56% / 85%) | TOWARD | — |
| DEN Broncos @ KC Chiefs — Moneyline | DEN Broncos (57% / 79%) | LIMITED / Held | KC Chiefs |
| CLE Browns @ JAX Jaguars — Spread | JAX Jaguars -7.5 (69% / 88%) | LIMITED / Held | CLE Browns +7.5 |
| SF 49ers @ LA Rams — Total | Under 48.5 (60% / 83%) | LIMITED / Held | Over 48.5 |
| ARI Cardinals @ LA Chargers — Total | Under 46.5 (58% / 94%) | LIMITED / Held | Over 46.5 |

Louisville/Ole Miss spread adds only the missing `Resistance Side` to Louisville +6.5; Ole Miss -6.5 already had `Pressure Side` under the prior rule.

## Ambiguous cases and controls

- ATL Falcons/PIT Steelers total returned to its opener through an active Whipsaw. Pressure is descriptive, but resistance is withheld until the reversal is recovered.
- Tommy McMillen moved modestly with pressure but below the directional confirmation threshold. It receives Pressure Side only; limited favorable movement is not resistance.
- Ordinary small juice drift does not create a directional lean. Held primary numbers can establish descriptive resistance, while action qualification remains unchanged.

Two evidence anchors change, both intentionally, with no directional lean: SF Giants +1.5 becomes the anchor against NY Mets spread pressure, and Under 7 becomes the anchor against NY Yankees/SD Padres total pressure.

## Invariance results

Frozen-snapshot comparison:

- Classification changes: 0
- Directional-lean changes: 0
- Board-rank changes: 0
- Scoring or score-input changes: 0
- Role-neutral chip colors remain identical to Public Pressure and One-Way
- TOWARD/AGAINST movement colors remain unchanged

## Market Guide changes

- Clarified that Directional Reads are primary classifications and that Freeze/Watch may remain descriptive.
- Updated Pressure Side from public-pressure-only wording to qualifying concentrated betting pressure.
- Clarified Freeze ownership, opposing resistance evidence, and the separate actionable lean gate.
- Replaced the Resistance Side definition with the approved wording.
- Expanded Held to cover moneyline prices.
- Made Late match the closing-window and observed-path implementation.
- Made Point, Juice, Price, and blank movement definitions describe measurable display behavior rather than classification thresholds.
- Added Thin, Feed Risk, and Split Risk definitions.
- Corrected Data Quality to distinguish incomplete market data from capped split risk.
- Added Market Rank (MKT) as selected-market rank rather than game rank.
