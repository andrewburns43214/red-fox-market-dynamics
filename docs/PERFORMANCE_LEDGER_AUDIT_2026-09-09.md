# Performance ledger production-history and freeze audit

Date: September 9, 2026  
Deployment status: **Attempted and fully rolled back**

## Recommendation

**NEEDS ATTENTION before another deployment.** The first controlled production invocation exposed an all-blank timestamp compatibility case. Production was rolled back under the deployment gate before the service wrote any classification. A local correction now forces even all-missing timestamp inputs into an explicitly UTC-aware series, but it has not been redeployed.

The September 8 production export supports a defensible backfill of 9 directional Supported Side rows, including 4 final `red_fox_favorite_v2` rows. No September 7 final-pregame classification archive was found, so no September 7 result is manufactured.

## Reconciliation of the previously reported evidence

The earlier September 8 review used files exported from production to:

`C:\Users\Andrew\AppData\Local\Temp\redfox-recent-results-audit-20260908`

Those files were outside the repository. The local checkout therefore correctly reported that its own `data/` and deck artifact folders did not contain them. Production's `live_recent.csv` is rolling (10 hours for final games, 8 hours for unresolved games), so its current copy has also aged the September 8 games out.

The retained export contained:

- `live_recent.csv`: frozen board classifications, source-side payloads, Favorite v2 fields, freeze times, and ESPN event IDs.
- `red_fox_favorite_tracking.csv`: qualification and fall-off lifecycle records with timestamps.
- `final_scores_history.csv`: the then-current score history.
- `anomaly_board.csv`: the captured live board used in the earlier operational review; it was not used to infer final classifications in this backfill.

Copies, hashes, production inventory, pregame source rows, result evidence, and generated ledgers are retained under `audit/performance_backfill_20260909/`.

## Read-only production inventory

| Artifact | Production coverage observed | Relevant contents | Backfill value |
|---|---|---|---|
| `snapshots.csv` | 810,841 rows; observations 2026-03-06 19:40 UTC through 2026-09-10 02:51 UTC | Raw side, open/current line, ticket/money split, scheduled start, source timestamp, some score fields | Proves the latest raw source timestamp and side/line/price at or before kickoff; does not contain final Supported Side or Favorite classification |
| `anomaly_events.csv` | 287,876 rows; 2026-09-01 03:58 UTC through 2026-09-10 02:41 UTC | Side paths, reads, line/price and split observations | Useful audit detail; does not contain final Supported Side or Favorite freeze |
| `anomaly_board.csv` | 227 current/future rows; starts 2026-09-10 through 2026-10-04 | Full current Market Read, Supported Side, Favorite fields and `market_sides` | Rolling current board, not a historical final-state archive |
| `live_recent.csv` | 49 rolling rows; starts 2026-09-09 17:10 UTC through 2026-09-10 02:10 UTC | Frozen classification plus live/final score and ESPN event ID | Complete for rows still inside retention; current production version does not yet have the new explicit source-state timestamp |
| `red_fox_favorite_tracking.csv` | 2,060 rows; recorded 2026-09-07 03:01 UTC through 2026-09-10 02:41 UTC | v1/v2 qualification state, side, line, read, rank, snapshot ID, timestamps | Reliable lifecycle evidence; insufficient alone to establish a final Supported Side/Favorite at kickoff |
| `final_scores_history.csv` | 368 rows; resolved 2026-09-01 through 2026-09-10 | Game ID, teams, final scores, resolution time | Reliable when the target game ID is present; current file did not retain the September 8 target IDs |
| `decision_freeze_ledger.csv` | 2,982 legacy rows; freezes 2026-04-03 through 2026-09-02 | Older engine decisions | Predates current Supported Side/Favorite v2 semantics and is excluded |
| `anomaly_action_ledger.csv` | 432 rows; captures 2026-09-01 through 2026-09-10 | Action candidates and observed lines | Does not establish the required final Supported Side/Favorite state and is excluded |

The production filesystem contained no archived copies of `live_recent.csv`. The saved September 8 production export is therefore the only retained final-classification source for those games.

## Historical validity and recovered coverage

- Current Supported Side semantics were introduced September 7, 2026 at 1:57:58 PM ET.
- `red_fox_favorite_v2`, requiring an exact confirmed Supported Side match, was introduced September 7, 2026 at 6:11:12 PM ET.
- Favorite performance is never accepted before the v2 boundary.
- No September 7 final-pregame classification record could be reconstructed from retained artifacts.
- The first recovered official row starts September 8, 2026 at 6:40 PM ET (22:40 UTC).

The backfill uses only the saved production `live_recent` classification, the matching latest raw production snapshot timestamp at or before kickoff, Favorite tracking through kickoff, and the retained ESPN event ID. It does not infer classifications from screenshots, the later board, narrative notes, or outcomes.

Recovered counts:

- Directional Supported Side rows: **9**
- Final Favorite v2 rows: **4**
- Unique games: **8**
- Graded rows: **9**
- Validation-only Supported Side result: **6 W / 3 L / 0 Push**
- Validation-only Favorite v2 result: **4 W / 0 L / 0 Push**

The earlier three completed Favorites reconcile exactly:

- NY Mets, moneyline, +113 opener to -105 final price, 31% tickets / 11% money, final Favorite v2, won 7-5.
- Houston Astros, moneyline, +129 opener to +114 final price, 22% tickets / 25% money, final Favorite v2, won 6-5.
- Los Angeles Angels, moneyline, +123 final price, final Favorite v2, won 6-1.

Texas Rangers was in progress in the earlier capture. Its retained final Favorite v2 state was TEX +104; ESPN event `401816865` later resolved 10-5, producing the fourth Favorite validation row.

## Freeze mechanism

The board export now adds `state_as_of_utc` for every market, derived from the latest synchronized source observation used for that market. The Live & Recent handoff writes:

- `final_pregame_state_at_utc`: the classification's source timestamp;
- `frozen_at_utc`: when the worker persisted the row;
- `freeze_method`: the provenance of the handoff.

The handoff accepts a row only when `state_as_of_utc <= scheduled_start`. The ledger independently repeats that validation, sorts candidate states by source timestamp, and takes the latest valid state at or before kickoff. A missing source timestamp or a post-kickoff source timestamp is rejected. Once the sport/event/market ledger key is written, later input cannot alter its classification fields. Result attachment is assertion-checked against the immutable classification columns.

The raw-snapshot recovery path remains available for the public live-score screen, but it is marked `raw_snapshot_recovery_no_classification`; because it has no Supported Side, it cannot enter the performance ledger.

## Market Read reporting

There is one combined directional ledger. Each row carries both `market_read` and `grade`. A CSV user can group directly by `market_read` to calculate Contrarian, Follow, Watch/Freeze-with-direction, or any future directional category. Non-directional rows without a Supported Side are excluded and receive no grade. No duplicate per-read ledgers are created.

## CSV auditability

The combined CSV contains event/game identity, sport, scheduled start, market, exact Supported Side, final side line/price, Market Read and detail, Favorite v2 status and lifecycle timestamps, source-state and persistence timestamps, final score, W/L/Push grade, score-provider identity, freeze provenance, snapshot ID, source hash, and audit status.

`closing_line`, `closing_price`, and `clv` remain blank unless a comparable close is explicitly retained. The September 8 backfill leaves them blank.

Representative row (abridged only for presentation):

```csv
event_id,game,sport,scheduled_start,market,supported_side,final_pregame_price,market_read,favorite_qualified,favorite_first_qualified_at,final_pregame_state_at_utc,final_pregame_frozen_at,final_score,grade,score_provider,score_provider_event_id,freeze_method
34635033,NY Mets @ MIA Marlins,mlb,2026-09-08T22:40:00+00:00,MONEYLINE,NY Mets,-105,Contrarian,yes,2026-09-08T00:51:49.355464+00:00,2026-09-08T22:30:55.672193+00:00,2026-09-08T22:40:04.731803+00:00,NY Mets 7 - MIA Marlins 5,W,espn,401816858,historical_production_export_plus_raw_snapshot_timestamp
```

## Performance and regression validation

- Maintained test suite: **461 passed, 17 subtests passed**.
- Focused ledger/freeze/live tests: **33 passed**.
- Explicit tests cover timer offsets, rejection of post-kickoff states, rejection of missing source timestamps, immutable classification, v1 exclusion, directional-only grading, and admin protection.
- Isolated ledger run over the 47-row retained file: median **69.38 ms**, maximum **100.62 ms** across eight clean runs.
- Per-market source timestamp propagation on a representative 240-market board: median **2.467 ms**, maximum **3.384 ms** across 100 runs.
- The performance service makes no odds or score API calls and runs outside snapshot, scoring, ranking, and publication processes. Outcome attachment consumes already-retained scores during separate maintenance.
- No scoring, ranking, Market Read, Supported Side, or Favorite rule changed.

Expected production impact is negligible. The only board-path addition is a small in-memory group-and-merge to persist source timestamps; measured cost is about 2.5 ms at 240 markets. Ledger reads and atomic writes occur in a separate service.

## Deployment gate

On September 9 at approximately 11:11 PM ET, the first production service invocation failed before writing because the rolling legacy `live_recent.csv` contained an all-blank `final_pregame_state_at_utc` column. Pandas represented that column without a timezone and rejected comparison to timezone-aware kickoff values.

The deployment was fully rolled back:

- performance timer disabled and removed;
- performance service removed;
- seed and export CSVs removed;
- prior Nginx configuration restored and validated;
- prior access verifier restored and restarted;
- Git deployment reverted by commit `b92c21a`;
- existing customer site and access verifier confirmed healthy.

The service failed before writing, so no ledger classification was created or altered. The local compatibility correction is covered by an explicit all-missing timestamp test; the maintained suite now reports **462 passed and 17 subtests passed**.

### Second controlled deployment

The revised service was tested against an isolated copy of production's 46 legacy rolling rows before enablement. The input had no `final_pregame_state_at_utc` column; the dry run completed successfully, ingested zero rows, used 69,324 KB maximum RSS, and took 1.05 seconds including Python startup. The seeded one-shot then completed successfully with 9 Supported Side rows and 4 Favorite v2 rows. All three unauthenticated production download routes returned HTTP 401.

The timer unit subsequently showed `active (elapsed)` with no next activation. Systemd had retained its September 9 11:11 PM ET trigger state from the first deployment, so the timer was enabled but was not scheduled prospectively. This failed the explicit timer gate. No timer configuration or other functional change was made. The second deployment was fully rolled back by production commit `bcbd5a5`; its timer/service, seed files, admin routes, and verifier changes were removed, and the customer site was confirmed healthy.
