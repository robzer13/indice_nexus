# I2_CANONICAL_COMPUTATION_V1.1 — DENOMINATOR RANGE COMPATIBILITY

**Status:** FROZEN — V1.1
**Base:** I2_CANONICAL_COMPUTATION V1.0 @ SHA256 fa5f6c10851182660c8b11526266620556cc01d5e8b42c769819cbbcf3d3ed35
**Formula change:** NO
**Scope:** deterministic handling of valuation ranges originating from a governed bounded ECONOMIC_SHARE_COUNT.

All V1.0 formulas, weights, anchors, caps and terminal-gate rules remain unchanged.

## 1. Existing range algebra retained

Expected-return numeric ranges continue to be scored at ordered endpoints.

RETURN_COMPONENT, OVS, INVESTMENT_RAW and INVESTMENT_SCORE continue to propagate their lower and upper endpoints using the existing monotonic formulas.

No midpoint, mean, distribution or endpoint selection is introduced.

## 2. Admission boundary

I2 does not decide whether a share-count bound is lawful. That decision belongs to OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0.

I2 receives only valuation outputs that have already passed the analytical denominator representation rule.

If MOS or VALUATION_RELIABILITY is NOT_ASSESSABLE, numeric OVS remains prohibited exactly as in V1.0.

## 3. Investment Class resolver

For numeric scalar Investment Score, retain the frozen bands.

For a numeric Investment Score range [L,H]:

- if class(L) == class(H), emit that same Investment Class;
- otherwise emit NOT_AVAILABLE and preserve the reason DENOMINATOR_RANGE_CROSSES_INVESTMENT_CLASS.

Bands remain:
90–100 EXCEPTIONAL;
80–<90 ATTRACTIVE;
70–<80 ADEQUATE;
50–<70 UNATTRACTIVE;
<50 POOR.

No favorable/unfavorable endpoint is selected.

## 4. Gate inputs

This patch does not invent or recompute analytical elite-gate judgments. An upstream denominator-sensitive predicate is supplied as PASS, FAIL or NOT_ASSESSABLE under the methodology authority's universal-feasible-set rule. Existing I2 terminal logic remains unchanged.

## 5. Precision

All range endpoints retain full calculation precision. Presentation rounding never becomes a calculation input.

## 6. Non-goals

No schema migration, database write, historical rewrite, OQS/OVS formula change, Investment Score formula change, policy-threshold change or terminal-gate change is authorized by this document.
