# I2 — Canonical computation and contract validation

I2 is the pure, deterministic analytical layer for OroTitan Equity Research V1. It neither reads nor writes a database. Production Supabase already contains the I1 identity foundation, but this increment performs no Supabase operation and adds no migration.

## Semantic and certification vocabulary

Canonical semantic values are `UNKNOWN`, `NOT_APPLICABLE`, `NOT_ASSESSABLE`, `MISSING`, and `NOT_AVAILABLE`. They are distinct values: zero remains a real score, while `null`, an empty string, and `NaN` are invalid.

Business research and investment conclusions use `CERTIFIED`, `CERTIFIED_WITH_LIMITATIONS`, `NOT_CERTIFIED`, or `INSUFFICIENT_DATA`. Score permission is `ALLOWED`, `CONDITIONAL`, or `SUSPENDED`; valuation reliability is `HIGH`, `MEDIUM`, `LOW`, or `NOT_ASSESSABLE`; and margin of safety is `ROBUST`, `ADEQUATE`, `THIN`, `NONE`, or `NOT_ASSESSABLE`.

## Exact formulas

For the seven numeric dimensions, in canonical order:

```text
OQS_RAW = 0.20 MOAT + 0.15 RUNWAY + 0.20 RETURN_QUALITY
        + 0.10 CASH_ECONOMICS + 0.15 CAPITAL_ALLOCATION
        + 0.10 MANAGEMENT_GOVERNANCE + 0.10 RESILIENCE_RISK
WEAK_LINK_CAP = min(100, min(all seven applicable scores) + 25)
OQS = min(OQS_RAW, WEAK_LINK_CAP)
```

No weights are renormalized. A nonnumeric dimension therefore makes this formula unavailable and validation fails closed. `NOT_CERTIFIED` business research, or `SUSPENDED` score permission, makes `OQS_RAW`, `WEAK_LINK_CAP`, and `OQS` `NOT_AVAILABLE`. Numeric dimension judgments are finite, in `[0,100]`, and divisible by 5; semantic states remain separate. OQS, OVS, and Investment Score results are not rounded to five-point increments.

Expected-return scoring linearly interpolates the anchors `(-10,0)`, `(-8,10)`, `(-6,20)`, `(-4,35)`, `(-2,50)`, `(0,70)`, `(2,82)`, `(4,90)`, `(6,95)`, `(8,100)`, saturating outside the end points. With five-year score `C` and ten-year score `N`:

```text
RETURN_COMPONENT = min(0.60 C + 0.40 N, N + 15)
OVS = min(RETURN_COMPONENT, MOS_CAP, VALUATION_RELIABILITY_CAP)
INVESTMENT_RAW = 0.70 OQS + 0.30 OVS
INVESTMENT_SCORE = min(INVESTMENT_RAW, OQS, OVS + 15)
```

MOS caps are 100/90/75/55 for `ROBUST`/`ADEQUATE`/`THIN`/`NONE`. Reliability caps are 100/95/80 for `HIGH`/`MEDIUM`/`LOW`; LOW remains explicitly limited. Either cap being `NOT_ASSESSABLE` prevents numeric OVS. A non-certifiable valuation prevents OVS and the investment score while leaving OQS independent.

The frozen I9 suspended-score rule is explicit and valuation-reliability driven: `SUSPENDED` with HIGH, MEDIUM, or LOW reliability produces OVS `NOT_AVAILABLE`, even when MOS is `NOT_ASSESSABLE`; `SUSPENDED` with `NOT_ASSESSABLE` reliability produces OVS `NOT_ASSESSABLE`. Numeric `INVESTMENT_RAW` and `INVESTMENT_SCORE` require numeric OQS and numeric OVS, with score permission permitting scoring; OVS is never substituted for unavailable OQS. Suspended investment score is always `NOT_AVAILABLE`.

## Evidence and terminal gate

MOAT and RUNWAY evidence uses `UNKNOWN`, `PLAUSIBLE`, `SUPPORTED`, `STRONGLY_SUPPORTED`, and `FALSIFIED`. `PLAUSIBLE` caps its dimension at 75 and `SUPPORTED` at 90. No unspecified ceilings are invented. Falsified evidence contradicts a positive corresponding elite judgment.

Canonical elite gates use only `PASS`, `FAIL`, and `NOT_ASSESSABLE`; no boolean collapse is used. OroTitan is `YES` exactly when business research and the investment conclusion are both `CERTIFIED`, score permission is `ALLOWED`, valuation reliability is `HIGH`, every canonical gate is `PASS`, and moat/runway `PASS` gates have `STRONGLY_SUPPORTED` evidence. `FAIL` or `NOT_ASSESSABLE` yields `NO`. `CERTIFIED_WITH_LIMITATIONS`, `CONDITIONAL`, and MEDIUM/LOW/NOT_ASSESSABLE reliability cannot produce `YES`. OQS never implies this terminal judgment and there is no proximity score.

## Stored and deterministic boundary

Stored, traceable judgments are dimension judgments, evidence, certification, elite gates, readiness, opportunity path, and next action. Recomputable values are OQS raw, weak-link cap, OQS, both interpolated return scores, return component, OVS, investment raw, investment score, and OroTitan status.

The contract accepts optional deterministic values only as assertions. It recomputes them and emits structured Zod issues for any mismatch; supplied values are never authoritative.

## Fail-closed validation and tests

The strict Zod schemas reject coercion, unknown keys, invalid enums, nonfinite or out-of-range scores, evidence-ceiling violations, contradictory elite gates, unavailable numeric scores, incomplete quality formulas, and deterministic mismatches. No value is clamped, rounded, repaired, or converted to a sentinel.

The computation and contract test suites cover the required C01–C53 matrix plus suspended/unavailable, I9 precedence, non-renormalization, five-point precision, evidence consistency, correct-recomputation, certification, and terminal mismatch cases. Floating-point assertions use an epsilon and calculations retain normal JavaScript precision.

## Explicit non-goals

I2 does not persist canonical snapshots, mutate Supabase, change schema or I1 identity tables, query production for writes, alter legacy scoring or historical snapshots, cut over an API/frontend/market price, backfill data, or change the frozen methodology. I3 remains responsible for persistence; I4 and I5 remain separate future increments.
