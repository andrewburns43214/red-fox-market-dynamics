# Snapshot capture/maintenance concurrency correction

## Scope and exact writer

Only the snapshot storage race is corrected. Scoring, ranking, Market Reads,
publication horizons, paired-side rules, freshness gates, capture/refresh
watchdogs, DK requests and timeout classification are unchanged.

The destructive writer was `main.update_snapshots_with_espn_finals()`, invoked
by `cmd_report_maintenance()` through the scheduled `run_maintenance.sh`.
It read the entire snapshots CSV, fetched/calculated finals, and wrote the old
dataframe back with `df.to_csv(src, index=False)`. Meanwhile `append_snapshot()`
appended captures to the live file. Their outer shell locks were different.
The old rewrite therefore removed captures appended after its initial read.
It also exposed a partly rewritten CSV to readers during its in-place write.

Production evidence from the prior verification showed successful capture
receipts through 04:20 UTC, followed by a file rewrite that reverted retained
timestamps to the 04:10–04:11 captures. That evidence remains in
`audit/coverage_fixes/production_20260906/capture_time_check.json`.

## Exact files changed

Production code:

- `main.py`: only `append_snapshot()` and
  `update_snapshots_with_espn_finals()` change. Captures delegate the completed
  batch to shared storage; maintenance saves its observed score values and
  commits only calculated final-score changes onto the latest history. Its log
  reports the number of changes actually persisted. Both use `SNAPSHOT_CSV`.
- `snapshot_store.py` (new): a common cross-process sidecar lock, atomic batch
  publication, and a streaming compare-and-set merge for final-score fields.

Verification and documentation:

- `tests/test_snapshot_concurrency.py` (new): 14 concurrency/storage regressions.
- `audit/snapshot_concurrency/verify_isolated.py` (new): exact baseline
  reproduction and production-sized immutable-copy verification.
- This report and generated evidence under `audit/snapshot_concurrency/`.

The unrelated pre-existing `audit_score_distribution.py` edit is untouched.
No shell scripts, cron configuration, network/auth configuration, scoring
modules or publication code were modified for this patch.

## Before and after

| Operation | Before | After |
|---|---|---|
| Capture while maintenance fetches/calculates | A later old-dataframe rewrite erased it | Capture proceeds; maintenance merges onto the latest stored rows |
| Simultaneous commits | Different locks; appending to a replaced inode could lose data | Both writers use the same persistent sidecar lock |
| Reader during a write | Could see a partial batch or truncated CSV | Sees the complete old or complete new file |
| Existing/extended CSV schema | Append assumed a fixed column order | Append respects the on-disk header, preserving extra metadata and finals columns |
| Stale concurrent finals proposal | Could overwrite a newer whole-file state | Updates only exact observed rows whose original score pair still matches |
| Write failure or killed writer | In-place rewrite could damage live history | Before atomic replace, the original remains intact; an interrupted process releases its OS lock |

ESPN calls and the existing final-score calculation stay outside the lock.
Maintenance creates a sparse proposal keyed by every observed non-score column
and both original scores. Under the lock, it streams the current CSV into a
temporary file, changing only matching final-score pairs. It retains row order,
duplicate observations, newly appended rows, unknown columns and newer finals.
Conflicting or incomplete identities fail safely; no fuzzy match is involved.

Capture builds the same rows and timestamp as before, copies existing history
bytes into a temporary file, appends the entire batch using the actual header,
then replaces the CSV atomically. Both paths fsync the completed file; Linux also
fsyncs the directory. Existing owner/group/mode are retained. The sidecar lock is
never removed or renamed. A 30-second lock-acquisition failure is explicit and
does not bypass the lock or change publication freshness limits.

These guarantees cover the two active scheduled snapshot writers. Legacy manual
rewrite/replay/purge scripts are not part of the scheduled pipeline and must not
be used concurrently against live storage; they were not migrated in this patch.

## Safe reproduction and historical preservation

The reproduction runs the exact baseline functions from revision `ecc388e` in
an isolated temporary data directory, with ESPN and history side effects stubbed.
The callback appends a valid two-sided capture after maintenance has read:

| Test | Immediately after append | After maintenance |
|---|---:|---:|
| Exact baseline implementation | 4 rows | 2 rows; concurrent pair lost |
| Corrected implementation | 4 rows | 4 rows; both histories retained |

The production-sized test used a copy taken while both existing production
writer locks were held. Tests never rewrote the live snapshot file.

- Original history: **478,445 rows**.
- After a two-row concurrent capture and a sparse finals merge: **478,447 rows**.
- **Every historical non-score field and row order preserved.**
- **Every unrelated historical final-score value preserved.**
- Only the two deliberately selected temporary test rows received score sentinels.
- Original copied input remained byte-for-byte unchanged.
- Source SHA-256: `ad8746aa24ac10edd9a38305cb10b8c8d0a1aace33bf4e755861ec1abec402a0`.
- Historical non-score/order digest:
  `8b01200815e55f0d9a2bf0a016db2adb2ad1712e07441944b7ea98d405f2ff9b`.
- Final measured capture commit: **0.49 seconds**; maintenance merge/commit:
  **16.25 seconds** on the production-sized copy under server load. These are
  observed timings, not a worst-case performance guarantee. Slow network/finals
  computation is excluded from lock occupancy; the streaming commit avoids a
  second full dataframe in the critical section.

## Regression and publication verification

Full maintained suite on Windows and isolated production Python 3.10/Linux:
**337 tests passed, 17 subtests passed**. `git diff --check` passes.

The new tests cover the actual maintenance entry point overlapping a capture,
independent processes in both commit orders, two processes creating the first
file, atomic reader visibility, process termination, replace failure, lock
timeout, duplicate observations, reordered/extended headers, stale finals,
conflicting proposals and incomplete identity rejection.

The integration test runs the actual publication pipeline after maintenance.
It verifies that the newest complete pair survives, remains fresh, is published
with both sides, and is the exact timestamp reported as the latest complete pair
in publication reconciliation. Historical rows and unrelated columns are
compared directly before and after. Existing horizon, pairing, scoring and Market
Read regression tests pass unchanged.

## Separate issues — deliberately not fixed here

- **DK cached response inconsistency:** prior same-time probes received
  HIT/STALE/MISS cache states with different market populations, including an
  empty cached page versus a populated fresh response. This needs a separate
  cache/pagination consistency patch and tests; no request or cache behavior
  changes are included here.
- **Timeout exclusion label:** a parsed capture killed by the shell watchdog can
  leave a RUNNING receipt and the default `RAW_MARKET_PARSE_FAILED` label. A
  separate change should record an explicit timeout reason and alert promptly.
  This patch does not change watchdogs, signal handling, receipt classification
  or monitoring rules.

## Rollout safety

Isolation, exact reproduction, full regression and production-sized preservation
checks are complete before any deployment. A safe rollout must quiesce both
existing outer writer locks so no already-running old maintenance process can
later execute its old rewrite. Then deploy only the scoped files, preserve a
snapshot/code backup, resume capture and verify historical prefix preservation
and fresh paired publication. No old-data restore should replace captures made
after a backup.
