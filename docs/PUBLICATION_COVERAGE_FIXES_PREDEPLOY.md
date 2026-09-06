# Publication coverage fixes — pre-deployment report

Implemented locally; **not deployed**. Production SSH still times out. No claim
is made that the live site is running these changes or that its current coverage
is verified. The unrelated pre-existing `audit_score_distribution.py` edit was
left alone.

## Production code changed

| File | Change |
|---|---|
| `main.py` | Bounded extreme-price moneyline identification; deterministic ESPN kickoff matching; explicit per-row validation states; verified DK unmatched rows retained; invalid identities/markets/kickoffs quarantined; scrape/capture receipts; rejected snapshots no longer purge an entire sport and its ledgers |
| `dk_headless.py` | Raw-page discovery observer before split filtering; enclosing event anchor preferred over unrelated preceding anchors; league-form evidence attached to rows; empty-render retries work; pagination/fetch/parse termination recorded |
| `dk_discovery.py` (new) | Independent event/market-header discovery, including unparseable markets; event/league evidence; raw kickoff extraction; shared by runtime and offline audit |
| `game_identity.py` (new) | Unicode/punctuation normalization, explicit scoped aliases, deterministic ordered game-pair matching, collision/ambiguity states; no fuzzy token-overlap matching |
| `publication_coverage.py` (new) | SQLite discovery/page/capture/publication journal, compressed raw-page archive, per-market exclusions and capture ages, run-versioned coverage counts, internal CSV/JSON summaries |
| `refresh_anomaly_board.py` | Stage-by-stage exclusion accounting; capture-validation quarantine; unchanged stale/pair/history gates; calendar-day DST arithmetic; empty-run accounting/freshness; explicit data-directory output routing |
| `coverage_monitor.py` (new) | Read-only freshness, census, capture/publication failure, interrupted-run, key-accounting, and export-hash checks; NFL/CFB counts and reasons |
| `run_all_sports.sh` | Runs coverage monitoring after refresh and logs coverage alerts |
| `ops/redfox-healthcheck.sh` | Includes coverage monitor failures in existing operational health status |

Also updated the offline audit to share header discovery, updated the old defect
characterization tests to the corrected behavior, and added runtime/identity tests.

`anomaly_board.py`, scoring modules, Market Read thresholds, rationale logic,
ranking logic, public board HTML, and the paired-side completeness function are
unchanged. Cross-source identity normalization is intentionally separate from
existing scoring/ledger side keys; this change does not migrate historical keys.

## Before versus after

| Defect | Before | After |
|---|---|---|
| Extreme moneylines | Favorite prices such as `-100000`/`-50000` did not match the three/four-digit odds pattern; opponent remained unpaired | Signed, integer American prices with magnitude 100 through 1,000,000 are recognized; malformed/out-of-range strings remain rejected; both sides still required |
| ESPN silent drops | A high batch match ratio caused unmatched DK games to be discarded | Source-verified, structurally identified DK rows survive with `ESPN_UNMATCHED` or `ESPN_UNAVAILABLE`; ambiguous/invalid identities have explicit quarantine reasons |
| Wrong-team matching | Token overlap could supply another game's kickoff | Exact deterministic name-pair lookup only; multiple event matches produce `ESPN_AMBIGUOUS` with no borrowed kickoff; valid independent DK evidence can remain captured |
| Freshness | Old rows vanished from publication with only aggregate logging | Same ten-minute active-price gate; discovered identity, latest capture, last complete pair, ages, and `STALE_CAPTURE` persist internally |
| Insufficient history | Evaluator returned no row | Threshold unchanged; omitted candidate market is recorded as `INSUFFICIENT_HISTORY` |
| CFB DST | 192 elapsed hours could end eight-calendar-date window one hour early | Local `DateOffset` calendar-day boundary; last intended local date remains included and ninth date remains excluded |
| NFL boundary arithmetic | Elapsed-day arithmetic could also drift across DST | Same Tuesday–Monday dates and bounded opening-week exception, calculated as local calendar dates |
| Whole-sport rejection | Could purge snapshots and historical state/ledgers | Rejected discovery is retained/quarantined and old histories preserved; rejected keys cannot enter the publisher |
| Capture/parse interruption | Could return partial results with no durable source census | Raw pages, discovered headers, run state, and termination reason survive; interrupted/incomplete evidence fails coverage health |
| Empty run | Could leave previous freshness metadata while board became empty | Explicit zero-market freshness plus a coverage report accounting for discovered-but-unpublished markets |

Freshness choice is deliberately conservative: **stale prices are not republished
as current**. No threshold was widened. An in-window market can still leave the
active board when stale, but its identity and exclusion are no longer lost. No
customer-facing stale-price placeholders or future-week listings were added.

## Names and identity

Deterministic cleanup removes apostrophe variants and Unicode combining marks,
normalizes punctuation/hyphens/spacing, and applies explicit aliases. Tests cover
Hawaii, Hawai'i, Hawai‘i, Hawaiʻi, HAW, Hawaii Rainbow Warriors, Nevada Las Vegas,
UNLV Rebels, State/St., directional abbreviations, LSU, UConn, Miami campuses,
and NFL shorthand such as KC Chiefs, LV Raiders, and NY Jets.

Directional schools, Nevada versus UNLV, Michigan versus Michigan State, and
Miami (FL) versus Miami (OH) remain distinct. Context-free ambiguous names such
as bare Miami, UH, USC, and mascot-only identities are flagged rather than
guessed. Unknown exact names can remain valid DK identities without acquiring an
unverified ESPN identity. A valid market does not require ESPN agreement merely
to stay captured and accounted for.

An invalid side identity is rejected without deleting valid other markets from
the same game. Retaining an incomplete capture does not publish it: the existing
same-timestamp, two-side, complete-field gates still apply.

Legacy auxiliary naming helpers in score-display/results/book-line paths remain
separate. This change removes fuzzy matching from the publication-validation
kickoff resolver; it does not claim to migrate every historical enrichment key
or alter results/rationale logic.

## Coverage results

Retained input: `data/release_rehearsal_rc/snapshots.csv`, SHA-256
`83499ad20019fd364cbc6f4e76c99e47f9854149115fb46f7552a5337238329c`.
Replay clock: September 3, 2026, 19:51:37.814387 UTC.

Both replays use the actual publication pipeline in isolated data directories.
The baseline substitutes the pre-change `infer_market_type` function from Git;
the other uses the fix. Results grading and action-ledger operations are stubbed
because they are unrelated to the publication comparison. These are not deployed
historical-run reconstructions or a claim of complete historical DK discovery.

| Retained in-window scope | Eligible scope | Published before | Published after | Explicit exclusions after |
|---|---:|---:|---:|---|
| CFB | 150 | 136 | **139** | 11 `STALE_CAPTURE` |
| NFL | 48 | 48 | **48** | None |
| All sports | 245 | 230 | **233** | 11 `STALE_CAPTURE`, 1 `INSUFFICIENT_HISTORY` |

**58 retained NFL future markets remain outside the publication window.** None
were added to the board. The three recovered markets are:

- Kent State @ South Carolina, moneyline, DK event `34180769`.
- Merrimack @ Delaware, moneyline, DK event `34502927`.
- Idaho @ Utah, moneyline, DK event `34502941`.

All 230 previously published market keys survive. Every common-market field is
identical except `board_rank`: 128 ordinal positions shift when three additional
rows enter the same unchanged ranking algorithm. Market Reads, explanations,
prices, side data, and all other common-market output fields are unchanged.
Evidence: `audit/coverage_fixes/replay_comparison.json`.

Saved DK source HTML from September 5 at approximately 11:11 PM Eastern was also
replayed through the new parser and validation with ESPN enrichment deliberately
not queried. All **210 NFL side rows and 26 CFB side rows** were retained, with
no validation rejections. These represent **48 NFL and 12 CFB in-window markets**.
Of 118 total NFL/CFB source markets, **58 remain intentional future exclusions**.
This verifies capture behavior on saved source evidence, not current production
publication. Evidence: `audit/coverage_fixes/current_source_validation.json`.

## Durable accounting and monitoring

Internal files under the configured data directory:

- `publication_coverage.sqlite3`: discovered identities, compressed raw pages,
  scrape/capture/publication runs, meaningful state transitions, historical run
  summaries. Interrupted processes leave visible `RUNNING` receipts.
- `publication_coverage.csv`: per-market current state, validation state, window
  scope, latest capture, latest complete pair, capture ages, eligibility and actual
  publication membership.
- `publication_coverage.json`: publication run ID, export SHA-256, expected active
  sports, source census health, NFL/CFB/all-sport counts and exclusion reasons.

The accounting denominator is supported, enabled, identified, **in-window pregame
scope**, before pairing/freshness/history failures. Gate-eligible counts are also
reported separately. Outside-window markets never increase missing-coverage
counts. For each league, scope count equals published plus explicit exclusions.

The existing Nginx configuration denies unallowlisted `/data/` files. No new public
route was added for these internal files, so the accounting does not expose future
DK games to customers.

Run `python coverage_monitor.py --data-dir data` for current counts and health.
It exits nonzero for missing/stale evidence, incomplete or old active-sport census,
stale in-window football markets, capture/publication failures, interrupted runs,
unexplained gaps, publication conflicts, or a mismatched export hash. Both the
runner and existing operational healthcheck invoke it. No external messages were
sent or alert-channel subscriptions changed.

Raw pages are compressed. Repeated publication clocks do not append another copy
of every unchanged historical market state. Page/run history still grows over
time; use the existing disk-health checks and establish an archive/retention policy
for long-term operation rather than deleting forensic evidence implicitly.

## Remaining limits and failure paths

- A completely changed DK page layout, missing event identity, failed request,
  pagination cap/repetition, or timeout can prevent a complete source census.
  These are explicit incomplete/failed states; they cannot certify zero missing
  games or claim discovery of games never delivered by DK.
- Publication or journal I/O failure cannot guarantee a new terminal record for
  that run. Failed/running receipts, missing/stale summary checks, and the board
  hash prevent it from being reported as a healthy reconciled run.
- A discovery with no accepted capture remains explicitly unparsed, quarantined,
  or capture-unavailable. Source disappearance alone is not treated as proof
  that DK withdrew the market.
- Pair semantics beyond the existing side-count/nonblank-field checks were not
  redesigned. The prior audit's semantic warning tests remain. No integrity gate
  was weakened to increase coverage.
- Browser search/date/market/signal filters and auth failures remain separate from
  publication coverage. The report proves exported membership, not the user's
  current browser filter state.
- Freshness remains a hard active-board gate. Coverage is explainable internally;
  this change does not promise every stale market stays visible to customers.
- Production source availability and deployed revision remain unverified until
  access is restored. No historical cause was assigned to UNLV–Hawaii.

For discovered keys in a successful instrumented publication run, all identified
filter paths now have terminal accounting. The system also reports incomplete
evidence rather than treating an unverified census as full coverage.

## UNLV–Hawaii follow-up

The expanded search normalized punctuation/apostrophes before looking for UNLV,
Nevada Las Vegas, Hawaii variants, Rainbow Warriors, and relevant abbreviations.
It searched 390 readable local artifacts. Seventeen files contained both schools
somewhere in the file, but row-level inspection found **no same-row matchup** under
the requested variants. Separate games in the same file are not matchup evidence.
See `audit/coverage_fixes/alias_history_search.json`.

Synthetic tests demonstrate that `Nevada Las Vegas @ Hawaii` matches ESPN
`UNLV Rebels @ Hawai‘i Rainbow Warriors` deterministically; ambiguous names never
borrow a kickoff. Runtime reconciliation tests demonstrate a discovered Hawaii
market progressing from explicit insufficient-history exclusion to publication
while a future market remains outside the window. These tests do not invent a
historical DK capture.

After production access is restored and deployment is reviewed, verify a current
real matchup using its DK event ID, raw-page receipt, validation state, capture,
complete pair, window decision, and published export hash.

## Verification

The full maintained `tests/` suite, including publication reconciliation and
scoring/board contracts, passed: **323 tests and 17 subtests**.
Both changed shell scripts pass `bash -n`; `git diff --check` passes.

The existing root-level one-off diagnostic scripts were not executed as a test
suite: some directly read/write operational artifacts. The maintained regression
suite is `python -m pytest tests`.

A replay initially reached the existing freshness writer's import-time default
directory. The call now passes `DATA` explicitly; the local generated freshness
artifact was corrected from its own local snapshots, and integration tests verify
isolated data-directory output. No production server files were changed.

Customer-visible changes after a future deployment: legitimate extreme-moneyline
markets and validated DK/ESPN-name-mismatch markets can become visible once all
existing gates pass; ambiguous/malformed inputs remain withheld; the last intended
CFB calendar hour is restored on DST transition weeks. Future weeks remain hidden,
stale active prices remain withheld, and no board UI, scoring, ranking algorithm,
or rationale change is included.
