# Scoring Spec v1

This document defines the plain-language meaning of the seven market-reaction patterns used by the engine:

- `FOLLOW`
- `FADE`
- `INITIATED`
- `BUYBACK`
- `STALE`
- `NOISE`
- `FREEZE`

The goal is to lock the semantic model before any more scoring changes are made.

Two rules apply everywhere in this spec:

1. A pattern must first answer "which side owns the signal?"
2. A pattern can describe interesting market behavior without being a bet.

If a pattern explanation sounds wrong when read aloud, the spec is wrong and code should not be changed until the spec is corrected.

## Global Terms

- `Pressure side`: the side attracting visible bets, money, or divergence-driven attention.
- `Opposite side`: the other side of the same market.
- `Book action`: what the market did in response to pressure.
- `Signal owner`: the side the market behavior actually belongs to.
- `Directional`: the pattern points to one side.
- `Non-directional`: the pattern describes market conditions but does not cleanly point to a side.
- `Directional conviction`: the market is expressing a view on a side through movement or resistance.
- `Directional price-opportunity`: a price mismatch exists that may be exploitable, but it does not by itself imply broad market conviction.
- `Non-directional descriptive`: the pattern describes market conditions without clean side ownership.
- `Meaningful pressure`: pressure strong enough to make book non-response informative. Baseline definition for v1 spec:
  - `bets_pct >= 60`, or
  - `money_pct >= 65`, or
  - `money_pct - bets_pct >= 15`
- `Key-number pinned`: a market is pinned to a key number if the current line value is within `0.5` of a defined key number and the market has not crossed it despite pressure. Initial examples:
  - spread keys: `3, 7, 10, 14`
  - totals: major round or half-step anchors used by the sport/book context

## FREEZE

`FREEZE` is not one thing. It must be decomposed before ownership, confidence, or decision logic are assigned.

### FREEZE_RESISTANCE

1. What market behavior created this pattern
Heavy or meaningful pressure appears on Side A, the book does not move in a meaningful way, and the market looks intentionally stable rather than stale or chaotic.

2. Which side owns the signal
The opposite side, Side B.

3. Does book action confirm, oppose, or merely describe that side
Oppose the pressure side and implicitly support the opposite side.

4. Is the pattern directional or non-directional
Directional.

Signal class
Directional conviction.

5. Is it actionable by default, conditionally actionable, or never actionable
Conditionally actionable.

6. What invalidates it
Evidence of stale pricing, key-number pinning, weak pressure, frequent reversals, or signs that money is actually balanced rather than one-sided.

7. What should it do to confidence score
Increase confidence on the opposite side if the hold looks deliberate and stable. It should not increase confidence on the public side.

8. What should it do to the decision
At most `LEAN` or `BET` on the opposite side if resistance is strong and stable. Never bet the pressure side because of resistance.

Concrete example
Seventy-one percent of bets and sixty-five percent of money are on Houston -2.5, and the book stays at -2.5 all day with no staleness signs. Read aloud: "The market kept taking Houston, and the book still refused to move off Illinois +2.5." That signal belongs to Illinois, not Houston.

### FREEZE_BALANCED

1. What market behavior created this pattern
Visible pressure exists, but the lack of movement is plausibly explained by balanced counteraction rather than resistance.

2. Which side owns the signal
No side by default.

3. Does book action confirm, oppose, or merely describe that side
Merely describes a balanced market.

4. Is the pattern directional or non-directional
Non-directional.

Signal class
Non-directional descriptive.

5. Is it actionable by default, conditionally actionable, or never actionable
Never actionable by itself.

6. What invalidates it
Clear evidence that the hold is actually resistance, staleness, or structural key-number behavior.

7. What should it do to confidence score
Low positive or neutral descriptive confidence only. It should not create strong side confidence.

8. What should it do to the decision
`NO_BET`.

Concrete example
Sixty-two percent of tickets are on a favorite, but respected money keeps showing on the dog and the line holds. Read aloud: "There is pressure, but the market is in balance." That is a market description, not a side pick.

### FREEZE_KEY_NUMBER

1. What market behavior created this pattern
Pressure appears, but the book holds a spread or total at a sticky key number where movement is structurally expensive.

2. Which side owns the signal
No side by default.

3. Does book action confirm, oppose, or merely describe that side
Merely describes market structure.

4. Is the pattern directional or non-directional
Non-directional.

Signal class
Non-directional descriptive.

5. Is it actionable by default, conditionally actionable, or never actionable
Never actionable by itself.

6. What invalidates it
A real off-key move, broad market confirmation, or evidence that the book is shading juice aggressively rather than simply protecting the number.

7. What should it do to confidence score
Suppress directional confidence. Structural holds should not be treated as conviction.

8. What should it do to the decision
`NO_BET`.

Concrete example
Public action piles onto a football favorite at -3, but the spread sits at -3 while juice changes around it. Read aloud: "The book may just be protecting 3." That is not enough to call the other side sharp.

### FREEZE_STALE

1. What market behavior created this pattern
One price appears frozen while the broader market has already moved.

2. Which side owns the signal
The side benefiting from the stale price, if cross-book comparison is trustworthy.

3. Does book action confirm, oppose, or merely describe that side
It does not confirm conviction; it exposes a stale booking opportunity.

4. Is the pattern directional or non-directional
Directional, but only in a stale-price sense.

Signal class
Directional price-opportunity.

5. Is it actionable by default, conditionally actionable, or never actionable
Conditionally actionable.

6. What invalidates it
Lack of cross-book confirmation, delayed feed issues, or a market that is not actually stale.

7. What should it do to confidence score
Raise stale-opportunity confidence, not broad market-conviction confidence.

8. What should it do to the decision
Potential `LEAN` or `BET` only if stale-book evidence is explicit. Otherwise `NO_BET`.

Concrete example
Consensus books move a total from 7 to 7.5, but one book still hangs 7. Read aloud: "This is stale Over 7, not a philosophical statement about the game." That is a price-opportunity signal, not generic freeze resistance.

### FREEZE_WEAK

1. What market behavior created this pattern
Pressure is present but too weak, too noisy, or too early for a no-move to carry meaning.

2. Which side owns the signal
No side.

3. Does book action confirm, oppose, or merely describe that side
Merely describes that nothing decisive has happened yet.

4. Is the pattern directional or non-directional
Non-directional.

Signal class
Non-directional descriptive.

5. Is it actionable by default, conditionally actionable, or never actionable
Never actionable.

6. What invalidates it
Stronger pressure, clearer persistence, or validated stale/resistance evidence.

7. What should it do to confidence score
Keep confidence low.

8. What should it do to the decision
`NO_BET`.

Concrete example
Fifty-eight percent of bets are on one side in the morning and the line has not moved yet. Read aloud: "That is mild attention, not meaningful resistance." No bet.

## FOLLOW

1. What market behavior created this pattern
Pressure forms on a side and the book moves meaningfully toward that same side.

2. Which side owns the signal
The displayed side itself.

3. Does book action confirm, oppose, or merely describe that side
Confirm.

4. Is the pattern directional or non-directional
Directional.

Signal class
Directional conviction.

5. Is it actionable by default, conditionally actionable, or never actionable
Conditionally actionable.

6. What invalidates it
Weak move size, late unstable movement, rapid reversals, or evidence that the move was just a stale correction.

7. What should it do to confidence score
Increase confidence if the move is meaningful and persistent.

8. What should it do to the decision
Can support `LEAN` or `BET` on the same side when persistence and edge are present.

Concrete example
Money keeps coming on a dog from +130 to +114 and the price holds. Read aloud: "The market moved toward the dog and stayed there." That belongs to the dog side.

## FADE

1. What market behavior created this pattern
Pressure forms on Side A and the book moves away from Side A.

2. Which side owns the signal
The opposite side, not the pressure side.

3. Does book action confirm, oppose, or merely describe that side
Oppose the pressure side and favor the opposite side.

4. Is the pattern directional or non-directional
Directional.

Signal class
Directional conviction.

5. Is it actionable by default, conditionally actionable, or never actionable
Conditionally actionable, but only on the opposite side.

6. What invalidates it
No real pressure, no real move, weak persistence, or classification that is actually a reversal or stale event.

7. What should it do to confidence score
Increase confidence on the opposite side and decrease confidence on the pressure side.

8. What should it do to the decision
Never recommend the pressure side. At most recommend the opposite side if the fade is clean and stable.

Concrete example
Public pounds Nebraska, but the spread moves toward Iowa. Read aloud: "The book moved against Nebraska pressure." That belongs to Iowa.

## INITIATED

1. What market behavior created this pattern
The book moves before there is clear public pressure.

2. Which side owns the signal
The side the move went toward.

3. Does book action confirm, oppose, or merely describe that side
Confirm the moved-toward side as an early market signal.

4. Is the pattern directional or non-directional
Directional.

Signal class
Directional conviction.

5. Is it actionable by default, conditionally actionable, or never actionable
Conditionally actionable.

6. What invalidates it
Later reversal, evidence of thin-open repricing, or proof that the move was noise rather than informed pressure.

7. What should it do to confidence score
Increase confidence, but less than a clean persistent `FOLLOW` unless persistence develops.

8. What should it do to the decision
Can support `LEAN` or `BET` if the move persists and survives later market interaction.

Concrete example
A side opens +350 and is quickly bet to +310 before public tickets build. Read aloud: "The market moved there before the crowd got involved." That is an initiated signal on the side that shortened.

## BUYBACK

1. What market behavior created this pattern
The market moved in one direction and then materially reversed after prior action.

2. Which side owns the signal
No side by default.

3. Does book action confirm, oppose, or merely describe that side
Merely describes a reversal battle unless a separate winner rule exists.

4. Is the pattern directional or non-directional
Non-directional by default.

Signal class
Non-directional descriptive.

5. Is it actionable by default, conditionally actionable, or never actionable
Never actionable by default.

6. What invalidates it
Clear dominance by one side after the reversal, in which case it should be reclassified under a more specific state rather than left as generic buyback.

7. What should it do to confidence score
Reduce confidence because the market message is unstable.

8. What should it do to the decision
`NO_BET` unless a future model explicitly identifies which side won the buyback battle.

Concrete example
A total jumps from 147.5 to 149.5 and then gets bet back to 147.5. Read aloud: "That market fought both ways." Until the winner is clear, no bet.

## STALE

1. What market behavior created this pattern
The book price lags an observable market move or looks behind relative to the broader market.

2. Which side owns the signal
The side getting the stale number.

3. Does book action confirm, oppose, or merely describe that side
Describe a price mismatch, not necessarily deep conviction.

4. Is the pattern directional or non-directional
Directional.

Signal class
Directional price-opportunity.

5. Is it actionable by default, conditionally actionable, or never actionable
Conditionally actionable.

6. What invalidates it
No external market confirmation, stale feed suspicion, or rapid correction proving the stale read was wrong.

7. What should it do to confidence score
Increase price-opportunity confidence only when the stale read is validated.

8. What should it do to the decision
Can support a bet only with explicit confirmation that the number is stale. Otherwise keep to `NO_BET` or light `LEAN`.

Concrete example
Most books move a hockey favorite from +130 to +114, but one feed still shows +130. Read aloud: "That is a stale +130." The opportunity belongs to the favorite at the stale number.

## NOISE

1. What market behavior created this pattern
Weak, mixed, contradictory, or low-information behavior with no clean directional read.

2. Which side owns the signal
No side.

3. Does book action confirm, oppose, or merely describe that side
Merely describes an unclear market.

4. Is the pattern directional or non-directional
Non-directional.

Signal class
Non-directional descriptive.

5. Is it actionable by default, conditionally actionable, or never actionable
Never actionable.

6. What invalidates it
A later clean directional state emerging.

7. What should it do to confidence score
Lower confidence.

8. What should it do to the decision
`NO_BET`.

Concrete example
Ticket share and money share are middling, line movement is tiny, and direction keeps wobbling. Read aloud: "There is no clean market story here." No bet.

## Decision Baseline Summary

- `FOLLOW`: directional, same-side, conditionally actionable
- `FADE`: directional, opposite-side, conditionally actionable
- `INITIATED`: directional, same-side, conditionally actionable
- `BUYBACK`: non-directional by default, not actionable
- `STALE`: directional only if externally validated, conditionally actionable
- `NOISE`: non-directional, not actionable
- `FREEZE_RESISTANCE`: directional conviction, opposite-side only, conditionally actionable
- `FREEZE_BALANCED`: non-directional, not actionable
- `FREEZE_KEY_NUMBER`: non-directional, not actionable
- `FREEZE_STALE`: directional price-opportunity, conditionally actionable
- `FREEZE_WEAK`: non-directional, not actionable

## KPI Readiness Rule

No KPI or calibration work should be considered trustworthy until:

- each pattern has fixed side ownership
- `FREEZE` is decomposed into subtypes
- confidence is separated from mere descriptiveness
- non-directional patterns cannot surface as directional bets
- regression fixtures exist for every spoken-aloud example in this document
