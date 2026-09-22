# OROTITAN_VNEXT_P0_VALUATION_ASSUMPTION_INTEGRITY_V0.1

**Project:** OroTitan Equity Research  
**Status:** CANDIDATE — GATE 15 / P0-7  
**Methodology change:** NO  
**Provider dependency:** NONE  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

This module implements Gate 15 / P0-7: Valuation Assumption Integrity.

It validates whether material valuation assumptions are traceable and economically consistent with frozen upstream research. It does not calculate intrinsic value, OVS, Investment Score, Valuation Reliability or Valuation Elite.

## 1. Basis matching

Frozen rule:

```text
FCFF -> WACC
FCFE / OWNER EARNINGS -> COST OF EQUITY
```

A sector-valid substitute must use its corresponding sector-valid discount basis.

If Owner Earnings are UNKNOWN, the module rejects a precise Owner Earnings DCF.

## 2. Fundamental forecast integrity

The forecast must be linked to the frozen driver chain:

```text
RUNWAY DRIVER
-> DENOMINATOR
-> PENETRATION / SHARE
-> UTILIZATION / WALLET
-> PRICE / MIX
-> REVENUE
-> OPERATING ECONOMICS
-> REINVESTMENT
-> FCF / OWNER EARNINGS
-> ECONOMIC SHARES
-> PER-SHARE CASH
```

A free-standing revenue CAGR without reconstructible driver support is rejected.

`RUNWAY_HORIZON` may not mechanically determine explicit forecast years.

## 3. Reinvestment consistency

Frozen relation:

```text
GROWTH ~= REINVESTMENT RATE x EXPECTED MARGINAL RETURN
```

The relevant return is expected marginal return / ROIIC, not historical ROIC.

A scenario combining high growth, high distributions and low capital needs requires explicit economic support.

## 4. Margin integrity

Material margin change must be causally decomposed through supported drivers such as price/mix, operating leverage, scale, product mix, input cost, competition and capacity.

The same margin expansion cannot be rewarded in forecast earnings and again through unsupported terminal valuation.

## 5. Share-count and financing integrity

Future economic shares must reconcile:

```text
OPENING SHARES
+ GROSS ISSUANCE
+ SBC / OPTIONS / RSUs
+ ACQUISITION SHARES
- BUYBACKS
= FUTURE ECONOMIC SHARES
```

Buybacks included in valuation require financing support from the upstream Financing Consistency artifact.

## 6. Terminal economics

Frozen relation:

```text
g = TERMINAL REINVESTMENT RATE x TERMINAL RETURN ON NEW CAPITAL
```

Terminal growth, margin and return cannot be independent optimistic assumptions.

An eroding moat cannot support permanent excess returns with no fade.

## 7. Discount-rate discipline

The economic discount rate must be reproducible and currency-consistent.

```text
ECONOMIC DISCOUNT RATE != INVESTOR REQUIRED RETURN
```

The former values cash flows. The latter remains a Price Ladder / investment-policy input.

## 8. Future M&A

Material future M&A value requires support from:

```text
target pool
deployable capital
organizational capacity
demonstrated acquisition returns
```

Historical acquired growth is not free future organic growth.

## 9. Optionality

Frozen Runway OPTIONALITY does not enter Base automatically.

The same optionality may not be included both in Base operating assumptions and again in terminal value.

## 10. Scenario discipline

Bear / Base / Bull scenarios must be causal rather than arbitrary percentage labels.

No default scenario probabilities are accepted without defended support.

## 11. Material Assumption Register

Every material assumption retains:

```text
ASSUMPTION_ID
VARIABLE
VALUE / RANGE
EPISTEMIC_TYPE
SOURCE / RATIONALE
SENSITIVITY
USED_IN
```

Assumption IDs are unique and exact.

Critical material assumptions may remain epistemically `UNKNOWN` only as an explicit limitation. They may not be fabricated into a point/range or silently converted to fact; downstream valuation certification must treat the unresolved critical input accordingly.

## 12. Assumption leakage / placeholders

An assumption may not silently become a fact in narrative.

Critical placeholders such as `TBD`, `XXX`, `[SOURCE]`, `N/A?`, `€?M` or explicit placeholder text block finalization.

A non-critical unresolved item must remain explicit UNKNOWN + limitation.

## 13. Authority boundary

The module may validate traceability, basis matching and cross-block consistency.

It must never:

- calculate intrinsic value;
- determine OVS / Investment Score;
- determine Valuation Reliability;
- determine Valuation Elite;
- invent missing assumptions;
- resolve UNKNOWN silently;
- write production.

## 14. Module Contract

```text
contracts/orotitan-equity/vnext/modules/
  VALUATION_ASSUMPTION_INTEGRITY.module-contract.v0.1.json
```

Dependencies:

```text
RETURN_NORMALIZATION
OWNER_CASH
FINANCING_CONSISTENCY
```

## 15. Runtime implementation

```text
runtime/vnext/modules/valuation-assumption-integrity.ts
```

No model-provider or Azure dependency exists.

## 16. Golden fixtures

```text
tests/fixtures/vnext/valuation-assumption-integrity.v0.1.json
```

Fixture classes:

```text
clean FCFF/WACC assumptions
FCFF / cost-of-equity mismatch
Owner Earnings UNKNOWN precise DCF
free-standing CAGR + mechanical runway horizon
growth / distribution / reinvestment conflict
margin + optionality double counting
terminal economics conflict
unsupported future M&A
assumption leakage + critical placeholder
unsupported default scenario probabilities
non-critical UNKNOWN placeholder
critical UNKNOWN material assumption
```

## 17. Gate 15 relationship

```text
P0-1 Weak Link Taxonomy              PASS / MERGED
P0-2 Decision State Architecture     PASS / MERGED
P0-3 Return Normalization            PASS / MERGED
P0-4 Capital Seasoning               PASS / MERGED
P0-5 Owner Cash                      PASS / MERGED
P0-6 Financing Consistency           PASS / MERGED
P0-7 Valuation Assumption Integrity  CANDIDATE
P0-8 Valuation Diagnostic Integrity  NOT STARTED
```

## 18. Acceptance matrix

```text
P0-7-01 basis matching preserved                                PASS
P0-7-02 Owner Earnings UNKNOWN firewall                         PASS
P0-7-03 driver-based forecast required                          PASS
P0-7-04 no mechanical runway-horizon forecast years             PASS
P0-7-05 growth/reinvestment/marginal-return consistency          PASS
P0-7-06 margin assumptions causal                                PASS
P0-7-07 margin terminal double-counting blocked                  PASS
P0-7-08 share-count bridge required                              PASS
P0-7-09 buybacks require financing support                      PASS
P0-7-10 terminal g/reinvestment/return consistency              PASS
P0-7-11 maturity / moat fade consistency                        PASS
P0-7-12 economic discount rate / investor hurdle separated      PASS
P0-7-13 future M&A support chain                                PASS
P0-7-14 optionality double-counting blocked                     PASS
P0-7-15 causal scenarios / no unsupported default probabilities PASS
P0-7-16 Material Assumption Register identity                   PASS
P0-7-17 assumption leakage blocked                              PASS
P0-7-18 critical placeholders blocked                           PASS
P0-7-18A critical assumption UNKNOWN preserved explicitly       PASS
P0-7-19 no valuation-score/reliability authority                PASS
P0-7-20 no provider/Azure dependency                            PASS
P0-7-21 deterministic CI                                        PENDING
```

## 19. Current state

```text
P0 MODULE = VALUATION_ASSUMPTION_INTEGRITY
STATUS    = CANDIDATE
LIVE MODEL REQUIRED = NO
SHADOW RUN MUTATION = FORBIDDEN UNTIL GATE 15
```
