# OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3 — FREEZE V3.0

**Status:** FROZEN — V3.0
**Base incorporated by reference:** OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2_FREEZE_V2.0 @ SHA256 3abfbccb0dca915f07af306b43657e73027bc40d6899d94e6af6d5e80fa3a293
**Depends on:** V3 Process + V3 Pilotage + unchanged V2 Research Stage
**Registry stage code:** DEEP_DIVE
**Methodology authority:** OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0
**Scoring formula change:** NO

All V2 Fundamentals and Certification rules remain unchanged except for the exact denominator-admission and propagation clauses below.

## 1. Fundamentals

Fundamentals still does not perform Valuation.

Where evidence supports it, Fundamentals may preserve exact denominator-relevant facts and hard constraints in the authoritative evidence/calculation/assumption lineage, but it may not decide a company-specific exception or use schema capability as methodology authority.

## 2. Valuation admission

Valuation requires, in addition to the V2 inputs, a denominator representation under the pinned global authority:

EXACT:
- exact point-in-time economic share count;
- SHARE_COUNT_AS_OF_DATE = REFERENCE_PRICE_DATE;
- complete exact state/bridge.

BOUNDED:
- finite positive LOWER_BOUND and UPPER_BOUND;
- BOUND_EFFECTIVE_DATE = REFERENCE_PRICE_DATE;
- BOUND_COMPLETENESS = COMPLETE;
- UNBOUNDED_MOVEMENT_CLASSES = 0;
- exact provenance/constraint references;
- correlation/state-conservation treatment complete.

UNKNOWN:
- VALUATION_ADMISSION = NO;
- no valuation model may start and no VALUATION_LOCK may be created.

## 3. Bound construction

Valuation may construct a denominator bound only by applying the global methodology to authoritative, cutoff-compliant evidence.

It must not use:
- midpoint or endpoint as a point estimate;
- stale exact denominator;
- weighted-average EPS denominator;
- vendor estimate;
- management expectation as a hard bound;
- unverified assumption of zero activity.

Every movement class must be exact or hard-bounded. Any unbounded relevant class -> UNKNOWN -> block.

## 4. Downstream propagation

A lawful bounded denominator must be propagated through:
- market capitalization / EV diagnostics;
- intrinsic value per share;
- primary expected return;
- Mature Normalization and permitted Same-Multiple diagnostic;
- Reverse DCF;
- Price Ladder;
- MOS;
- valuation reliability;
- I2 valuation inputs.

Use the joint feasible set when variables are correlated. Do not silently take a Cartesian product of marginal intervals.

MOS or another categorical valuation state is emitted only when all feasible denominator states map to the same frozen category; otherwise use the global authority's NOT_ASSESSABLE semantics.

## 5. VALUATION_LOCK

A BOUNDED representation may produce a VALUATION_LOCK only when every mandatory propagated output has a deterministic governed representation and the lock retains:
- denominator representation;
- share_count_as_of_date;
- bound provenance/constraint refs;
- all range endpoints at full calculation precision;
- any denominator-caused NOT_ASSESSABLE state;
- READY_FOR_CERTIFICATION state.

## 6. Certification

Certification applies unchanged formulas and gates. It must preserve I2 range outputs and the global universal-feasible-set gate semantics.

No score or gate may be improved by choosing a favorable denominator endpoint. No unfavorable endpoint may be selected as a substitute either.

## 7. Reopening and versioning

Any change to denominator evidence or constraints that changes the feasible set creates new immutable analytical artifact versions and invalidates dependent Valuation outputs under the existing V2 reopening protocol.

Historical runs pinned to an earlier Contract Set are never rebound to V3.
