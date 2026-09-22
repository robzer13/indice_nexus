# OROTITAN_VNEXT_P0_VALUATION_DIAGNOSTIC_INTEGRITY_V0.1

**Project:** OroTitan Equity Research  
**Status:** CANDIDATE — GATE 15 / P0-8  
**Methodology change:** NO  
**Provider dependency:** NONE  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

This module implements Gate 15 / P0-8: Valuation Diagnostic Integrity.

It validates valuation diagnostics and deterministic arithmetic without changing OVS anchors, Investment Score, Valuation Reliability methodology or OroTitan terminal logic.

## 1. Mature-normalization precedence

Frozen execution rule:

```text
1. valid numeric MATURE_NORMALIZATION_RETURN
   -> selected N basis

2. mature normalization legitimately NOT_ASSESSABLE / NOT_AVAILABLE
   + valid numeric SAME_MULTIPLE_RETURN
   -> Same-Multiple fallback

3. otherwise
   -> NOT_AVAILABLE / NOT_ASSESSABLE
   -> numeric OVS prohibited
```

When both diagnostics are valid and numeric:

```text
MATURE_NORMALIZATION_RETURN
> deterministic precedence over
NO_MULTIPLE_EXPANSION_RETURN
```

No MIN, MAX or AVERAGE selection is permitted.

## 2. Invalid Mature Normalization fails closed

```text
INVALID MATURE NORMALIZATION
!=
LEGITIMATE FALLBACK CONDITION
```

A material calculation failure, definition failure, basis mismatch or unresolved critical assumption in Mature Normalization cannot be bypassed through Same-Multiple.

## 3. Reverse DCF

Reverse DCF must:

```text
lock most fundamental variables
solve one material variable
translate the result into business economics
compare it with frozen fundamentals
```

Underdetermined multi-variable solving is rejected.

## 4. Expected return

Shareholder expected return uses modeled cash flows:

```text
t0: - entry price
t1...tn: + distributions
tn: + terminal share value
```

Exact IRR is required when interim cash flows are modeled.

Naive arithmetic addition of growth + distributions + valuation change is prohibited.

5Y is produced when defensible. 10Y is produced only when the economic horizon supports it.

## 5. Same-Multiple and Mature-Normalization

Same-Multiple remains a diagnostic / cross-check.

Mature Normalization is economically preferred because it asks whether the investment still produces strong returns without relying on an aggressive terminal valuation state.

## 6. Normalized multiple cross-check

Historical and peer multiples are cross-checks only.

If used, comparability must be established across:

```text
growth
margin
return
leverage
business mix
accounting
```

## 7. Cyclicals

When cyclical normalization is triggered, valuation must use mid-cycle earnings / FCF and through-cycle capital rather than peak-cycle economics.

## 8. Margin of Safety

The module does not decide the MOS state.

It verifies that the required input set is represented, including as applicable:

```text
central value
bear value
expected return
same-multiple / mature-normalization return
hurdle headroom
terminal dependence
sensitivity
valuation reliability
```

## 9. Price Ladder

The Price Ladder must:

```text
use the same frozen fundamentals
+
use configured required-return levels
```

The analytical return levels are policy/configuration inputs, not universal economic truths.

## 10. Sensitivity

When valuation is assessable, identify the 2-4 variables dominating value.

V0.1 does not reward giant sensitivity matrices.

## 11. Valuation Reliability

Canonical state remains:

```text
HIGH
MEDIUM
LOW
NOT_ASSESSABLE
```

This module preserves the supplied state but does not determine it.

`NOT_ASSESSABLE` remains a legitimate outcome and prevents numeric OVS through the diagnostic gate.

## 12. Mathematical integrity

V0.1 can require deterministic checks including:

```text
Market Cap = Price x Economic Shares
EV bridge
terminal revenue bridge
terminal metric bridge
PV explicit + PV terminal = value
equity bridge
value/share
exact IRR
```

and monotonicity:

```text
discount rate up -> DCF down
terminal growth up -> DCF up
cash up -> equity value up
debt up -> equity value down
```

A required material math check that is absent or not PASS blocks finalization.

## 13. Authority boundary

The module must never:

- change OVS formula / anchors;
- calculate or override Investment Score;
- determine Valuation Elite;
- invent a Mature-Normalization result;
- use Same-Multiple to hide an invalid Mature calculation;
- select MIN/MAX/AVERAGE of N diagnostics;
- manually override a score;
- write production.

## 14. Module Contract

```text
contracts/orotitan-equity/vnext/modules/
  VALUATION_DIAGNOSTIC_INTEGRITY.module-contract.v0.1.json
```

Dependency:

```text
VALUATION_ASSUMPTION_INTEGRITY
```

## 15. Runtime implementation

```text
runtime/vnext/modules/valuation-diagnostic-integrity.ts
```

No model-provider or Azure dependency exists.

## 16. Golden fixtures

```text
tests/fixtures/vnext/valuation-diagnostic-integrity.v0.1.json
```

Fixture classes:

```text
both numeric -> Mature precedence
legitimate Mature N/A -> Same-Multiple fallback
invalid Mature -> fail closed
no numeric N diagnostic
underdetermined Reverse DCF
invalid expected-return arithmetic
unsupported 10Y horizon
normalized-multiple comparability failure
cyclical without mid-cycle normalization
valuation monotonicity failure
excessive sensitivity variables
Valuation Reliability NOT_ASSESSABLE
```

## 17. Gate 15 relationship

```text
P0-1 Weak Link Taxonomy              PASS / MERGED
P0-2 Decision State Architecture     PASS / MERGED
P0-3 Return Normalization            PASS / MERGED
P0-4 Capital Seasoning               PASS / MERGED
P0-5 Owner Cash                      PASS / MERGED
P0-6 Financing Consistency           PASS / MERGED
P0-7 Valuation Assumption Integrity  PASS / MERGED
P0-8 Valuation Diagnostic Integrity  CANDIDATE
```

## 18. Acceptance matrix

```text
P0-8-01 Mature Normalization precedence                       PASS
P0-8-02 legitimate Same-Multiple fallback                     PASS
P0-8-03 invalid Mature fails closed                           PASS
P0-8-04 no MIN/MAX/AVERAGE N selection                        PASS
P0-8-05 non-numeric N state preserved                         PASS
P0-8-06 Reverse DCF one-variable discipline                   PASS
P0-8-07 Reverse DCF business translation                      PASS
P0-8-08 exact expected-return IRR                              PASS
P0-8-09 no naive expected-return addition                     PASS
P0-8-10 5Y/10Y horizon discipline                             PASS
P0-8-11 normalized-multiple comparability                     PASS
P0-8-12 cyclical mid-cycle normalization                      PASS
P0-8-13 MOS input integrity                                   PASS
P0-8-14 Price Ladder same fundamentals/configuration          PASS
P0-8-15 2-4 dominant sensitivity variables                    PASS
P0-8-16 Valuation Reliability NOT_ASSESSABLE preserved        PASS
P0-8-17 mathematical reconciliation checks                    PASS
P0-8-18 monotonicity checks                                   PASS
P0-8-19 no OVS/Investment Score/Elite authority               PASS
P0-8-20 no provider/Azure dependency                          PASS
P0-8-21 deterministic CI                                      PENDING
```

## 19. Current state

```text
P0 MODULE = VALUATION_DIAGNOSTIC_INTEGRITY
STATUS    = CANDIDATE
LIVE MODEL REQUIRED = NO
SHADOW RUN MUTATION = FORBIDDEN UNTIL GATE 15
```
