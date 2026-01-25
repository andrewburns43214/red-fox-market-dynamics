# Signal Ledger Hygiene + Migration (2026-01-24)

## Why this exists
Older runs produced THRESHOLD_CROSS rows with non-adjacent transitions (e.g. NO_BET -> STRONG_BET).
We now require adjacency-only transitions for audits.

## What was changed
1) Runtime behavior (main.py):
- THRESHOLD_CROSS logging emits adjacent step transitions only (NO_BET <-> LEAN <-> BET <-> STRONG_BET)
- logic_version hygiene forces blank/v? to LOGIC_VERSION on read (instrumentation-only)
- bucket hygiene normalizes from_bucket/to_bucket values (instrumentation-only)

2) One-time historical migration:
- Script: tools/ledger_migrations/fix_signal_ledger_expand_jumps_csv.py
- Expands historical jump transitions into adjacent steps (does NOT change non-THRESHOLD_CROSS rows)

A migration note is stored in:
- data/signal_ledger_MIGRATION_NOTE_20260124.txt

## How to verify integrity at any time
Run:
- python tools/ledger_check.py

Expected:
- bad_transition_count=0
- no v? logic_version rows
- tail rows match active logic version

## Important repo hygiene
We do NOT commit data/*.csv.
