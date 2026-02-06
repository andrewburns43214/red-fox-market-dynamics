# LOGIC_SPEC_v1.1_CANONICAL
logic_version: v1.1
design_locked: true

DO NOT MODIFY THIS FILE.
THIS FILE DEFINES THE SHIPPED LOGIC CONTRACT.

========================================
TIMING (SSOT)
========================================
EARLY = game day → kickoff −8h
MID   = −8h → −60m
LATE  = −60m → kickoff

Timing is computed ONCE and read first.
LATE = confirm / invalidate only.
No retroactive early logic.

========================================
SCORE
========================================
Score math is unchanged from v1.0.
All v1.1 discipline is LABEL-ONLY.

========================================
STEP 5 / STEP 4 — PERSISTENCE & DECAY (GLOBAL)
========================================
• Downgrades override upgrades
• No escalation after decay
• Peak → last compression blocks escalation
• LATE: no new upgrades
• Score is NEVER changed here

========================================
STEP 1 — STRONG CERTIFICATION
========================================
STRONG requires:
• score ≥ 72
• persistence met
• stability met
• timing allows (MID only)
• sport allows
• no blocks (public drift, cross-market)

STRONG_BET may NEVER appear without eligibility=true.

========================================
SPORT DAMPENERS
========================================
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
========================================
• No NaNs
• Schema stable
• Repeated runs do not flap labels
• STRONG rarity diagnostic only (no tuning)

========================================
END OF SPEC
========================================
