# LOGIC\_SPEC\_v1.1\_CANONICAL

logic\_version: v1.1
design\_locked: true

DO NOT MODIFY THIS FILE.
THIS FILE DEFINES THE SHIPPED LOGIC CONTRACT.

========================================
TIMING (SSOT)
===

EARLY = game day → kickoff −8h
MID   = −8h → −60m
LATE  = −60m → kickoff

Timing is computed ONCE and read first.
LATE = confirm / invalidate only.
No retroactive early logic.

========================================
SCORE
===

Score math is unchanged from v1.0.
All v1.1 discipline is LABEL-ONLY.

========================================
STEP 5 / STEP 4 — PERSISTENCE \& DECAY (GLOBAL)
===

• Downgrades override upgrades
• No escalation after decay
• Peak → last compression blocks escalation
• LATE: no new upgrades
• Score is NEVER changed here

========================================
STEP 1 — STRONG CERTIFICATION
===

STRONG requires:
• score ≥ 72
• persistence met
• stability met
• timing allows (MID only)
• sport allows
• no blocks (public drift, cross-market)

STRONG\_BET may NEVER appear without eligibility=true.

========================================
SPORT DAMPENERS
===

NCAAB:
• EARLY dampener
• single-market dependency penalty
• LATE = confirmation only
• tighter compression tolerance

NCAAF:
• lighter EARLY dampener
• single-market penalty
• compression dampener

========================================
INVARIANTS
===

• No NaNs
• Schema stable
• Repeated runs do not flap labels
• STRONG rarity diagnostic only (no tuning)

========================================
========================================
===

# STEP 1 — STRONG CERTIFICATION (COMPLETE)

# ========================================

# STRONG requires ALL of the following:

# • score ≥ 72

# • persistence satisfied (≥2 ticks; ≥3 for NCAAB)

# • stability satisfied (no peak → last decay)

# • NOT in LATE window

# • NCAAB: never STRONG in LATE

# • Public Drift blocks STRONG

# • Cross-market contradiction blocks STRONG

# • Only ONE STRONG side per game/market

# • STRONG is certification, not a score change

# 

# ========================================

# STEP 2 — SPORT DAMPENERS

# ========================================

# NCAAB:

# • Early dampener

# • Single-market dependency penalty

# • Late confirmation-only

# • Compression blocks escalation

# 

# NCAAF:

# • Lighter versions of above

# • No volume caps

# 

# MLB:

# • Included in v1.1 with MINIMAL governors only

# • See MLB section below

# 

# ========================================

# STEP 3 — TIMING DISCIPLINE (ORDER OF OPS)

# ========================================

# Timing is evaluated FIRST.

# No early logic may execute after timing is known.

# 

# EARLY:

# • Sanitation only

# • No upgrades allowed

# 

# MID:

# • Normal certification allowed

# • Trajectory validation allowed

# 

# LATE:

# • Confirm / invalidate only

# • NO upgrades

# • NO new STRONG

# 

# ========================================

# STEP 4 — PERSISTENCE \& STABILITY

# ========================================

# • Downgrades override upgrades

# • No escalation after decay

# • Peak → last compression blocks escalation

# • Applies to ALL buckets

# • Score is NEVER changed here

# 

# ========================================

# STEP 5 — THRESHOLDS \& BUCKETS

# ========================================

# LEAN   ≥ 60

# BET    ≥ 68

# STRONG ≥ 72 (certification-gated)

# 

# Thresholds are frozen.

# Buckets do not alter scores.

# 

# ========================================

# METRICS (INSTRUMENTATION ONLY)

# ========================================

# • row\_state.csv stores last/peak state

# • signal\_ledger.csv logs threshold events

# • logic\_version tags every write

# • Metrics NEVER affect live logic

# • v1.1 activation is explicit, not implicit

# 

# ========================================

# MLB v1.1 MINIMAL GOVERNORS

# ========================================

# ONLY the following are allowed:

# 

# • Starting Pitcher (SP) confirmation gate

# &nbsp; – Unconfirmed SP → cap at BET

# 

# • Late SP scratch

# &nbsp; – Auto-block STRONG

# 

# • Basic rest governor

# &nbsp; – Day game after night → soft cap (STRONG → BET)

# 

# Explicitly excluded:

# • Weather

# • Park factors

# • Lineup math

# • Refactors

# • New data sources

# 

# ========================================

# NON-GOALS / DO NOT TOUCH

# ========================================

# • Scoring math

# • Threshold values

# • Snapshot ingestion

# • DK kickoff as SSOT

# • ESPN kickoff overrides

# • Metrics influencing logic

# • Large refactors

# 

# ========================================

# ENFORCEMENT

# ========================================

# This document is CANONICAL.

# Code must conform to this spec.

# Spec changes require explicit versioning.

