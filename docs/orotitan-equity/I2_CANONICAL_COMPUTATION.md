# I2 — Canonical computation and contract validation

I2 is the pure, deterministic analytical layer for OroTitan Equity Research V1. It neither reads nor writes a database. Production Supabase already contains the I1 identity foundation, but this increment performs no Supabase operation and adds no migration.

## Semantic and certification vocabulary

Canonical semantic values are `UNKNOWN`, `NOT_APPLICABLE`, `NOT_ASSESSABLE`, `MISSING`, and `NOT_AVAILABLE`. They are distinct values: zero remains a real score, while `null`, an empty string, and `NaN` are invalid. Numeric dimensions may also be bounded ranges `{ min, max }`; range endpoints are ordered, finite, in `[0,100]`, and use five-point increments. Canonical computed score ranges use ordered finite endpoints in `[0,100]` and are never rounded.

Business research and investment conclusions use `CERTIFIED`, `CERTIFIED_WITH_LIMITATIONS`, `NOT_CERTIFIED`, or `INSUFFICIENT_DATA`. Score permission is `ALLOWED`, `CONDITIONAL`, or `SUSPENDED`; valuation reliability is `HIGH`, `MEDIUM`, `LOW`, or `NOT_ASSESSABLE`; and margin of safety is `ROBUST`, `ADEQUATE`, `THIN`, `NONE`, or `NOT_ASSESSABLE`.

## Investment policy V1

The deterministic execution layer recognizes exactly the locked policy:

```text
POLICY_VERSION
= OROTITAN_INVESTMENT_POLICY_V1.0.0

REQUIRED_RETURN_H
= 10.0%

STRONG_RETURN_THRESHOLD
= 12.5%

EXCEPTIONAL_RETURN_THRESHOLD
= 15.0%
```

These are execution-policy inputs, not replacements for the frozen economic discount-rate doctrine. I2 never derives, optimizes, or overrides them. I3-B rejects an analysis-capable Phase-4 snapshot whose Price Ladder does not carry these exact V1 values.

## Exact formulas

For the seven numeric dimensions, in canonical order:

```text
OQS_RAW = 0.20 MOAT + 0.15 RUNWAY + 0.20 RETURN_QUALITY
        + 0.10 CASH_ECONOMICS + 0.15 CAPITAL_ALLOCATION
        + 0.10 MANAGEMENT_GOVERNANCE + 0.10 RESILIENCE_RISK
WEAK_LINK_CAP = min(100, min(all seven applicable scores) + 25)
OQS = min(OQS_RAW, WEAK_LINK_CAP)
```

No weights are renormalized. A nonnumeric dimension therefore makes this formula unavailable and validation fails closed. For bounded dimensions, OQS is propagated monotonically: all lower endpoints produce the lower result and all upper endpoints produce the upper result, without enumerating combinations. Scalar-only inputs preserve scalar outputs. `NOT_CERTIFIED` business research leaves valid numeric/range `OQS_RAW` and `WEAK_LINK_CAP` computed but publishes `OQS` as `NOT_AVAILABLE`; `SUSPENDED` score permission makes all three unavailable. Semantic states remain separate.

Expected-return scoring linearly interpolates the anchors `(-10,0)`, `(-8,10)`, `(-6,20)`, `(-4,35)`, `(-2,50)`, `(0,70)`, `(2,82)`, `(4,90)`, `(6,95)`, `(8,100)`, saturating outside the end points. `C` is the score of `PRIMARY_EXPECTED_RETURN`.

`N` is selected deterministically before scoring by `selectNormalizationReturn`:

```text
IF valid numeric MATURE_NORMALIZATION_RETURN exists
→ use it as N basis

ELSE IF MATURE_NORMALIZATION_RETURN is legitimately
NOT_ASSESSABLE / NOT_AVAILABLE after required work
AND valid numeric NO_MULTIPLE_EXPANSION_RETURN exists
→ use NO_MULTIPLE_EXPANSION_RETURN

ELSE
→ preserve the applicable Mature-Normalization unavailable semantic state
→ numeric OVS prohibited

INVALID / UNRECONCILED MATURE_NORMALIZATION_RETURN
→ FAIL CLOSED
→ no fallback
```

When both returns are valid and numeric, Mature Normalization always has precedence. The selector is not `MIN`, `MAX`, or analyst discretion and therefore selects Mature even when it is numerically above Same-Multiple and also when it is below Same-Multiple. The Same-Multiple return remains available in the Phase-4 payload as a diagnostic/cross-check; it becomes the N basis only through the legitimate fallback branch above.

A structurally invalid Mature value fails immediately. A valid numeric Mature value whose downstream deterministic outputs do not reconcile also fails closed through canonical I2 reconciliation; the presence of a valid Same-Multiple number does not authorize bypassing that failure.

```text
RETURN_COMPONENT = min(0.60 C + 0.40 N, N + 15)
OVS = min(RETURN_COMPONENT, MOS_CAP, VALUATION_RELIABILITY_CAP)
INVESTMENT_RAW = 0.70 OQS + 0.30 OVS
INVESTMENT_SCORE = min(INVESTMENT_RAW, OQS, OVS + 15)
```

Expected-return delta ranges are scored at their ordered endpoints. `RETURN_COMPONENT`, `OVS`, and Investment Score ranges use the same lower/upper endpoint propagation because each operation is monotonic. Fixed MOS and valuation-reliability caps apply independently to both OVS endpoints.

MOS caps are 100/90/75/55 for `ROBUST`/`ADEQUATE`/`THIN`/`NONE`. Reliability caps are 100/95/80 for `HIGH`/`MEDIUM`/`LOW`; LOW remains explicitly limited. Either cap being `NOT_ASSESSABLE` prevents numeric OVS. A non-certifiable valuation prevents OVS and the investment score while leaving OQS independent.

The frozen I9 suspended-score rule is explicit and valuation-reliability driven: `SUSPENDED` with HIGH, MEDIUM, or LOW reliability produces OVS `NOT_AVAILABLE`, even when MOS is `NOT_ASSESSABLE`; `SUSPENDED` with `NOT_ASSESSABLE` reliability produces OVS `NOT_ASSESSABLE`. Numeric `INVESTMENT_RAW` and `INVESTMENT_SCORE` require numeric OQS and numeric OVS, with score permission permitting scoring; OVS is never substituted for unavailable OQS. Suspended investment score is always `NOT_AVAILABLE`.

## Evidence and terminal gate

MOAT evidence uses `UNKNOWN`, `PLAUSIBLE`, `SUPPORTED`, `STRONGLY_SUPPORTED`, and `FALSIFIED`. RUNWAY evidence uses `UNKNOWN`, `PLAUSIBLE`, `SUPPORTED`, and `STRONGLY_SUPPORTED`; `FALSIFIED` is structurally invalid for RUNWAY. `PLAUSIBLE` caps its dimension at 75 and `SUPPORTED` at 90. No unspecified ceilings are invented. Falsified MOAT evidence contradicts a positive corresponding elite judgment.

Canonical elite gates use only `PASS`, `FAIL`, and `NOT_ASSESSABLE`; no boolean collapse is used. OroTitan is `YES` exactly when business research and the investment conclusion are both `CERTIFIED`, score permission is `ALLOWED`, valuation reliability is `HIGH`, every canonical gate is `PASS`, and moat/runway `PASS` gates have `STRONGLY_SUPPORTED` evidence. `FAIL` or `NOT_ASSESSABLE` yields `NO`. `CERTIFIED_WITH_LIMITATIONS`, `CONDITIONAL`, and MEDIUM/LOW/NOT_ASSESSABLE reliability cannot produce `YES`. OQS never implies this terminal judgment and there is no proximity score.

The N-selector patch does not modify the terminal gate. A changed valuation output can naturally change a separately stored valuation-gate judgment in a future analytical dossier, but no terminal boolean rule is changed by I2.

## Stored and deterministic boundary

Stored, traceable judgments are dimension judgments, evidence, certification, elite gates, readiness, opportunity path, and next action. Recomputable values are OQS raw, weak-link cap, OQS, both interpolated return scores, return component, OVS, investment raw, investment score, and OroTitan status.

The contract accepts optional deterministic values only as assertions. It recomputes them and emits structured Zod issues for any mismatch; supplied values are never authoritative.

## Fail-closed validation and tests

The strict Zod schemas reject coercion, unknown keys, invalid enums, nonfinite or out-of-range scores, evidence-ceiling violations, contradictory elite gates, unavailable numeric scores, incomplete quality formulas, and deterministic mismatches. No value is clamped, rounded, repaired, or converted to a sentinel.

The computation and contract test suites cover the required matrix plus scalar/range propagation, expected-return saturation, OVS and investment caps, suspended/unavailable semantics, I9 precedence, non-renormalization, five-point precision, distinct evidence vocabularies, evidence consistency, correct-recomputation, certification, terminal mismatch cases, exact investment-policy values, Mature-over-Same-Multiple precedence in both relative-value directions, legitimate fallback, semantic-state preservation, and invalid Mature fail-closed behavior. Floating-point assertions use an epsilon and calculations retain normal JavaScript precision.

## Explicit non-goals

I2 does not persist canonical snapshots, mutate Supabase, change Phase-4 schema shape or I1 identity tables, query production for writes, alter legacy scoring or historical snapshots, cut over an API/frontend/market price, backfill data, change OQS/OVS/Investment Score formulas, change the terminal gate, or reopen frozen methodology. I3 remains responsible for persistence admission; the N selector is deterministic execution semantics under `OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1`.
