# Publication and exclusion audit — September 5, 2026

Production behavior is unchanged. This audit adds only offline inspection code,
regression tests, and evidence files. Scoring, Market Reads, thresholds, ranking,
pair requirements, freshness thresholds, and publication horizons were not edited.
The existing unrelated edit to `audit_score_distribution.py` was left alone.

## Findings and evidence limits

Coverage cannot currently be certified. The pipeline has no durable source-market
inventory joined to publication outcomes. Several filters log aggregate row counts,
but not the identities and reasons of excluded markets. A successful run therefore
does not establish complete coverage.

There is a confirmed normalization defect in retained captures: moneyline inference
accepts only three- or four-digit odds. Five/six-digit favorites are captured but
lose their market classification; their opposing sides then fail pairing. This is
an upstream parsing defect, not a reason to relax the pairing gate.

Production SSH to the known host timed out. The public production CSV endpoints
returned HTTP 401 in the preceding investigation. No authenticated browser surface
was available. The running revision, environment overrides, cron timing, production
snapshots, and contemporaneous published export could not be verified. Unknown
production counts below are **not zeros**. Local exports were not represented as
current production data.

## Counting rules

Grain is `(sport, game_id, market_display)`, not sides, raw rows, or whole games.

1. **Discovered:** a market header observed in raw DK HTML, independently of whether
   its split bars or odds parse. Unidentifiable event/market records remain explicit
   investigation records; they cannot certify a market census.
2. **In-window scope:** supported league/market, enabled season, inside the existing
   publication horizon, with kickoff not expired. This is the denominator for
   source coverage; bad capture or missing pairing must not shrink it.
3. **Publication-gate eligible:** in-window scope plus the current capture,
   normalization, complete-pair, freshness, kickoff, and evaluator-history gates.
4. **Published:** an observed matching key in the export from the same run. A
   publication-ready key absent from an aligned export is an
   `UNEXPLAINED_PUBLICATION_GAP`, which fails the audit assertion.

Both scope coverage and gate-eligible coverage should be reported. Reporting only
the latter would conceal upstream loss by redefining the denominator.

## Current DK observation

Captured September 5 at approximately **11:11 PM Eastern** (September 6,
03:11 UTC). Raw HTML, URLs, per-page timestamps, hashes, parser counts, and
pagination stop reasons are retained in `audit/coverage_20260905/source/`.

| League | Discovered markets | Outside intentional horizon | In-window scope | Production published |
|---|---:|---:|---:|---|
| NFL | 105 | 57 | 48 | Unknown |
| CFB | 13 | 1 | 12 | Unknown |
| **NFL + CFB** | **118** | **58** | **60** | **Unknown** |

The 58 outside-window markets are `OUTSIDE_PUBLICATION_WINDOW`, never missing
coverage. Current production gate eligibility cannot be determined without its
capture history; do not equate the 60 scope markets with 60 proven ready markets.

Across all seven configured sports queried: 200 market headers were discovered.
Besides the football markets there were 40 MLB, 25 UFC, and 17 NHL markets.
The NHL season switch is off on September 5, so those 17 are
`SPORT_DISABLED_BY_SEASON`. NBA and CBB returned explicit no-events responses.
This leaves 125 currently in-scope markets across enabled sports, all with
`CAPTURE_EVIDENCE_UNAVAILABLE` for the production comparison.

All 200 discovered header keys matched a market key from the existing DOM parser
on these downloaded pages. This does not establish successful production capture,
paired-side completeness, or complete upstream coverage. Repeated terminal pages
and navigation are recorded; the independent HTTP observation is deliberately
marked `source_census_complete=false` rather than certify server-side completeness.

## Retained capture replay

Input: `data/release_rehearsal_rc/snapshots.csv`, 287,835 side rows. SHA-256:
`83499ad20019fd364cbc6f4e76c99e47f9854149115fb46f7552a5337238329c`.
Replay clock: September 3, **19:51:37.814387 UTC**, the last retained capture.
This applies the current local gates to older evidence; it is not a reconstruction
of the deployed September 3 code or an aligned production publication run.

The inventory is a lower bound inferred from retained captures, not a historical
DK census. Wide-odds sides are identified as moneyline in the audit inventory only;
the pipeline replay still applies unchanged production inference.

| Mutually exclusive replay disposition | Markets |
|---|---:|
| Kickoff expired | 939 |
| Outside intentional publication window | 58 |
| In-window, no capture in two-hour working set | 11 |
| In-window, moneyline normalization failure causing missing pair | 3 |
| In-window, insufficient parseable paired history | 1 |
| Pass current gates; aligned publication evidence unavailable | 230 |
| **Total retained market identities** | **1,242** |

Of the 245 in-window scope markets, 230 pass the current gates and 15 have the
explicit exclusions above. All other audited per-market gate reasons have zero
first exclusions in this replay, **not zero incidents in production**.

For football specifically: 256 retained markets, 58 outside the horizon, 198
in-window, 184 gate-eligible. CFB accounts for 150 in-window markets: 136 ready,
11 outside the working capture set, and 3 normalization/pair failures. NFL has
48 ready in-window markets. Production published counts remain unknown.

The 11 CFB capture-window exclusions are Indiana State–Purdue (2 markets),
ULM–Mississippi State (3), Northern Illinois–Iowa (3), and Florida Atlantic–Florida
(3). This evidence establishes their last captured state, not whether DK had
withdrawn them or the collector/validator subsequently missed them.

The three current normalization/pair failures are:

| Game | Captured favorite price | Last retained capture UTC |
|---|---|---|
| Kent State @ South Carolina | South Carolina @ -100000 | Sept. 3, 19:43:27.934788 |
| Idaho @ Utah | Utah @ -100000 | Sept. 3, 19:43:27.934788 |
| Merrimack @ Delaware | Delaware @ -50000 | Sept. 3, 19:43:27.934788 |

Both sides exist in the same capture, but `main.infer_market_type()` fails the
favorite's odds width. The audit gives these markets
`MONEYLINE_ODDS_WIDTH_NORMALIZATION_FAILURE`. Northern Illinois–Iowa also has
the defect in older captures, but its first current exclusion is the two-hour
working-set gate; it is not double-counted.

The one history exclusion is BOS Red Sox @ BAL Orioles moneyline. The evaluator
has fewer than two parseable paired observations. No pair-value warning or
one-sided evaluator survival was found in this replay.

## Every exclusion and loss path found

Numbers reference the evidence above; “unknown” means operational evidence was
not available. Aggregate failures cannot honestly be assigned per-market counts
without the missing source manifest.

| Stage / path | Current behavior and machine-readable audit reason | Measured impact |
|---|---|---|
| Runner season selection | Sport never collected outside configured dates; `SPORT_DISABLED_BY_SEASON` | 17 current NHL headers intentionally out of scope |
| Overlapping run / watchdog | `flock` exits; each snapshot has a 120-second default timeout; refresh has a 300-second default timeout; errors continue to the next stage | Production market impact unknown |
| HTTP/render failure | Empty response, import/browser failure, swallowed exception, or page failure stops/returns no records; no per-market inventory | Unknown |
| Pagination | 20-page cap; empty/repeated page stops; earlier records can still be returned as success after a later-page failure | Zero unmatched header keys on observed pages; undiscovered remainder not certified |
| Sport identity | Fast HTTP path trusts requested sport; every parsed row is assigned it. Nonempty wrong/mixed HTML bypasses form-based fallback | No form mismatch in current captured pages; production unknown |
| DOM discovery/parser | Requires progress-bar containers, two usable percentages; missing structure skips rows. Event lookup walks backward up to 60 anchors | Zero observed header-to-parser market-key gaps now; historical unknown |
| Deduplication | Key uses sport/game ID/market/side; missing identity or normalization collisions can merge records | Unknown |
| No parsed names / no rows | Empty scrape returns early; no names fails validation | Unknown |
| ESPN validation | With at least 85% matches, unmatched DK games are discarded. At 35–85%, full slate survives only if fallback passes; below 35%, fallback can reject the whole sport | Reproduced: 1 valid DK game dropped in a 10-game synthetic slate; production count unknown |
| Fallback quality gates | Batch thresholds for IDs, lines, market types, kickoff coverage and rows/game can reject a slate | Unknown |
| UFC validation | Cross-sport overlap >=50% or non-moneyline share >15% rejects slate | Unknown |
| Whole-sport purge | Zero rows after failed validation invokes purge of snapshots and five other live/state files, with backups; anomaly export later loses candidates | Unknown; especially important for historical evidence retention |
| Persistence | Snapshot timestamp assigned by Red Fox at append; invalid timezone string becomes empty; file/open-registry writes can fail | Unknown |
| Market inference | Unsupported/unrecognized inferred market removed before history; wide moneylines lose one side | 3 active failures, 4 games showing defect across retained history |
| Timestamp / working set | Invalid capture timestamps dropped; only rows within 2h of newest supported capture across sports considered | 11 in-window markets |
| Same-capture pair | Requires >=2 distinct normalized sides at one timestamp, chooses newest such timestamp, dedups side rows | 3 failures explained by wide-odds inference; no additional unexplained pair failures in replay |
| Complete fields | Exactly 2 side keys and 2 nonblank essential rows required; no fallback to older complete fields after selecting the newest pair | 0 additional first exclusions in replay |
| Freshness | Capture older than 10 minutes removed; exact boundary retained; configurable environment threshold | 0 at base replay clock; 136 CFB exclusions at +3 minutes |
| Kickoff | Missing/invalid kickoff excluded; cutoff is strictly greater than now minus 5 minutes | 939 expired retained identities; 0 additional missing-kickoff first exclusions |
| Football horizon | CFB local rolling window; NFL Tuesday–Monday with bounded opening-week exception | 58 current and 58 historical intentional exclusions, different source inventories |
| Evaluator history | Unsupported market or fewer than 2 parseable paired-history points returns `None` per side | 1 additional in-window market |
| Leader selection | One row per existing market; consolidates sides rather than intentionally dropping whole market keys | No whole-market loss established; no scoring changed |
| Export/runtime | Exceptions/timeouts before replacement leave old export; files replaced individually rather than as one versioned bundle | Unknown; must distinguish publication failure from intentional exclusion |
| Empty export/freshness | Header-only board is valid; freshness writer returns without updating if no source rows remain | Unknown; stale metadata can survive an empty board |
| Browser retrieval | HTTP/auth/CSV failure becomes null and then an empty board | Production audit got 401; user-session state unknown |
| Browser filters | Sport, market, date, signal, search, priority views hide rows; UFC non-moneyline removed client-side | Actual user filter state unknown |
| NFL browser filter | Separate first-upcoming-kickoff +7-day filter and immediate past-kickoff removal differs from backend Tuesday–Monday and five-minute grace | No additional week leakage/loss proved; boundary behavior differs |
| Live & Recent | Separate handoff/10-hour retention path and scoreboard calls; not proof of pregame publication | UNLV–Hawaii handoff not available |

Relevant implementation: `dk_headless.py`, `main.py` (`validate_snapshot_rows`,
`infer_market_type`, `append_snapshot`, `purge_sport_from_live_files`),
`refresh_anomaly_board.py`, `anomaly_board.py` (`_evaluate_side`,
`_build_history_points`), `run_all_sports.sh`, `site/board.html`,
`build_live_recent.py`, and `ops/redfox-healthcheck.sh`.

## Freshness and boundary conclusions

The ten-minute age is **Red Fox capture age**, not a DK last-change timestamp.
Successfully recapturing unchanged odds refreshes it. Calling unchanged prices
stale solely because they did not move would misdiagnose the current system.

The documented polling cadence is ten minutes, equal to the default cutoff.
Serial sports, scrape variation, a missed run, or a partial latest pair can use
up that margin. A fresh single side does not refresh an older complete pair.
Replay at 19:54:37.814387 UTC leaves 94 gate-ready markets instead of 230:
all 136 previously-ready CFB markets become `STALE_CAPTURE`. Their last paired
capture was 19:43:27.934788 UTC. This proves the deterministic removal mechanism;
it does not prove the production cron encountered that exact delay.

Safest recommendation: retain stale complete pairs internally with their original
capture timestamp and exact exclusion reason; alert on acquisition/validation
failure. Do not silently republish them as fresh or widen the threshold. A later
product decision could expose an explicitly unavailable/stale game status without
active signal eligibility; this audit does not implement it. First measure the
full-run cadence and complete-pair age before proposing any threshold change.

CFB uses Eastern midnight inclusive through eight days later exclusive. NFL uses
Tuesday–Monday; September 1–14, 2026 has the existing bounded September 9–14
opening-slate exception. Exact edges and the Hawaii late-night UTC conversion are
tested. No September 5 horizon defect explaining UNLV–Hawaii was found.

Technical boundary defect: `Timedelta(days=8)` is 192 elapsed hours, not eight
local calendar dates. On the fall-back transition, the endpoint can be one hour
early: November 8 at 23:30 Eastern is excluded from the November 1 eight-date
window. NFL's elapsed-day arithmetic also warrants DST boundary tests. Recommend
local calendar-date arithmetic preserving the intended dates, not a wider horizon.
Yearless DK kickoff parsing also depends on the runtime year with only a
December-to-January rollover exception; raw historical replay needs capture time.

The complete-pair function enforces side count and nonblank fields, not all semantic
consistency. A synthetic mismatched total passes it; the audit flags the disagreement.
Recommend separate semantic checks (opposing identities, paired line values,
numeric percentage ranges, same kickoff) without loosening existing requirements.
No such pair-value anomaly was found in the retained replay.

## UNLV–Hawaii reconstruction

No matchup record was found in the readable local raw HTML, CSV/JSON histories,
ledgers, or top-level exports searched. The September 1–3 CFB captures in both
staging/rehearsal copies contain no UNLV/Hawaii-named CFB game. The fresh August 31
CFB raw capture also has no matching team name. See
`audit/coverage_20260905/unlv_hawaii_search.json` for search scope and count.

At the live source audit time (11:11 PM Eastern), DK's observed CFB pages no longer
included the matchup. Its expected 10 PM Eastern kickoff would already be beyond
the five-minute pregame cutoff, but that cannot explain its absence earlier.
The synthetic Hawaii test verifies time conversion only; it is not historical
evidence of this game's DK kickoff.

Last known captured state: **not recoverable from available evidence**.
Exact earlier exclusion: **unknown**. Do not label it stale, ESPN-unmatched,
outside-window, or never offered without the missing raw/log evidence. Production
raw captures, validation logs/backups, and run-aligned snapshots/exports are needed.

## Minimal fixes proposed, not deployed

1. Add an append-only discovery manifest before any parser/validation gate. Record
   run ID, page URL/hash, league evidence, event/market identity, kickoff source,
   capture time, and pagination completeness. Retain failures and unknown headers.
2. Record per-market stage transitions and exact first exclusion plus diagnostic
   secondary reasons. Preserve `OUTSIDE_PUBLICATION_WINDOW` as intentional and
   out of the coverage denominator. Retain prior discoveries when they vanish
   from a response; absence alone is not proof of source withdrawal.
3. Correct wide-odds market identification with bounded numeric validation,
   preferably using the source market header. Keep two-sided completeness intact.
4. Audit unmatched ESPN games individually; use ESPN as corroborating identity/time
   evidence rather than silently discarding verified DK games at a batch ratio.
   Quarantine ambiguous identity with a reason. Do not bypass league validation.
5. Record `INSUFFICIENT_PARSEABLE_PAIRED_HISTORY` while retaining discovered games
   internally. Do not lower the evaluator's observation threshold.
6. Reconcile to a versioned export from the same run, then atomically publish its
   audit summary. A gate-ready absent key must fail coverage health, even if the
   export command succeeded. Keep public row presentation/ranking unchanged.
7. Add per-sport complete-capture age and empty-export markers. Existing health
   checks focus on file age and log errors, not eligible-market key loss.
8. Repair DST calendar arithmetic separately, with unchanged intended dates and
   tests, after review. Align browser/backend boundary contracts explicitly.

## Tests and operational acceptance

Added offline modules `audit/publication_coverage.py` and
`audit/capture_dk_inventory.py`; neither is wired into the runner. Reports preserve
unknown evidence rather than outputting a false 100% coverage figure.

Regression suite: **38 tests passed** across the new coverage tests and existing
snapshot synchronization / DK parser tests. It covers source headers without split
bars, unverified leagues, missing evidence, explicit exclusion accounting,
unexplained publication gaps, no input mutation, stale exact boundaries, newer
partial snapshots, both football horizons, NFL rollover/opening exception,
Hawaii kickoff/grace boundary, DST defect characterization, ESPN unmatched loss,
wide-odds normalization loss, and paired-total semantic warning.

Before enabling production certification, add integration fixtures for failed
pages/empty render retries/pagination caps, unmatched ESPN decisions tied to raw
keys, normalization collisions, invalid timestamps, all semantic pair failures,
export interruption/empty export, browser filter state and authenticated retrieval,
and full-run timing near freshness expiry. Test two captures of an unchanged
market and one new market's transition from insufficient history to publication.

Required monitoring: discovered counts and census completeness by league; intentional
future exclusions; in-window scope; gate-ready and published key counts; exclusions
by reason and duration; previously-published eligible keys lost this run;
complete-pair capture age; parser/validation rejection rates; and publication run
IDs. Alert immediately on unexplained eligible-key loss or incomplete census,
without changing scoring or expanding future-week publication.

Reproduce read-only analysis with:

```powershell
.\.venv\Scripts\python.exe -m audit.publication_coverage --inventory audit/coverage_20260905/source/inventory.csv --as-of 2026-09-06T03:11:20Z --output audit/coverage_20260905/current_source_reconciliation
.\.venv\Scripts\python.exe -m audit.publication_coverage --snapshots data/release_rehearsal_rc/snapshots.csv --as-of 2026-09-03T19:51:37.814387Z --output audit/coverage_20260905/historical_replay
.\.venv\Scripts\python.exe -m pytest tests/test_publication_coverage.py tests/test_refresh_snapshot_sync.py tests/test_dk_headless.py -q -p no:cacheprovider --basetemp .pytest-coverage-audit-new
```

For an actual source-to-production comparison, supply `--inventory`, `--snapshots`,
and `--board` from the **same captured publication run** plus its clock and deployed
configuration. Current offline conclusions do not certify that missing comparison.
