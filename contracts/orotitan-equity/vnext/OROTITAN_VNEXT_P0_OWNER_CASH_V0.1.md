# OROTITAN_VNEXT_P0_OWNER_CASH_V0.1

**Project:** OroTitan Equity Research  
**Status:** CANDIDATE — GATE 15 / P0-5  
**Methodology change:** NO  
**Provider dependency:** NONE  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

This module implements the fifth Gate 15 P0 analytical component: Owner Cash.

It operationalizes the frozen cash architecture without creating a new cash score or modifying Cash Economics weights.

```text
ACCOUNTING EARNINGS
↓
OPERATING CASH GENERATION
↓
REQUIRED OPERATING REINVESTMENT
↓
STANDARDIZED FCF
↓
OWNER EARNINGS
↓
PER-SHARE ECONOMIC CASH
```

## 1. Frozen cash distinctions

V0.1 preserves:

```text
NET INCOME != CASH
CFO != FCF
COMPANY-REPORTED FCF != STANDARDIZED FCF
STANDARDIZED FCF != OWNER EARNINGS
NON-CASH != NON-ECONOMIC
RECURRING ONE-OFF != ONE-OFF
```

Reported FCF, Standardized FCF and Owner Earnings may not be silently collapsed into one measure.

## 2. Standardized FCF

For ordinary non-financials the frozen base remains:

```text
STANDARDIZED_FCF
=
CFO
-
CASH_CAPEX
```

The module validates the selected architecture but does not recompute ledger arithmetic itself.

Standardized FCF must not be overloaded to manufacture Owner Earnings.

## 3. Owner Earnings

Frozen concept:

```text
OWNER_EARNINGS
=
NORMALIZED OPERATING CASH GENERATION
-
MAINTENANCE REINVESTMENT
```

Owner Earnings may be:

```text
POINT
RANGE
UNKNOWN
```

for the industrial framework.

A POINT estimate is rejected when a material maintenance component is unresolved or unknown.

A RANGE is allowed to preserve explicit maintenance uncertainty.

UNKNOWN remains a valid outcome when the maintenance economics cannot be supported.

## 4. Maintenance components

V0.1 can trace:

```text
PHYSICAL_CAPEX
INTANGIBLE_INVESTMENT
WORKING_CAPITAL
LEASE_CASH
RECURRING_RESTRUCTURING
ECONOMIC_CASH_TAXES
OTHER
```

For each component it keeps materiality and whether treatment is resolved.

The module does not infer maintenance capex from D&A.

```text
D&A != MAINTENANCE CAPEX
```

True maintenance investment remains an analytical judgment supported by evidence.

## 5. Double-counting firewalls

Frozen controls implemented deterministically:

```text
TOTAL CASH CAPEX already in Standardized FCF
-> do not subtract the same maintenance cash again from that FCF measure

WORKING CAPITAL already in CFO
-> do not subtract the same movement again

SBC
-> one economic cost, one penalty
```

Owner Earnings remains conceptually derived from normalized operating cash generation less maintenance reinvestment, not by mechanically taking Standardized FCF and subtracting maintenance capex a second time.

## 6. Working capital

Structural negative working capital may be a genuine business-model advantage.

But:

```text
temporary stretched payables
inventory liquidation
one-off aggressive collection
!=
structural Owner Earnings
```

V0.1 rejects temporary working-capital benefits when they are presented as structural owner cash.

## 7. Deferred maintenance

Material deferred maintenance must be reflected in the Owner Cash conclusion.

A temporary reduction in capex cannot be accepted as sustainable owner cash when catch-up investment is materially required.

## 8. Recurring adjustments

Recurring restructuring, transformation or acquisition-related costs remain economic when they are a recurring cost of the business model unless supported evidence establishes otherwise.

V0.1 rejects a material recurring-adjustment burden that is removed from economic cash without support.

## 9. SBC and dilution

Frozen invariant:

```text
SBC = ECONOMIC COMPENSATION COST
```

Central rule:

```text
ONE ECONOMIC SBC COST
-> ONE PENALTY
```

For material SBC, V0.1 requires:

```text
economic cost recognized = YES
sbcPenaltyCount = 1
```

It rejects both zero recognition and double penalty.

## 10. Acquisitions

V0.1 preserves:

```text
ORGANIC CASH ECONOMICS
!=
ACQUISITION CAPITAL ALLOCATION
```

Acquisitions are not automatically subtracted from Standardized FCF.

When acquisition capital is material, organic cash economics must be separated from acquisition deployment.

## 11. Per-share cash

If per-share economic cash is produced, the denominator must use economic diluted shares rather than a convenient basic share count.

## 12. Sector-valid substitutes

For banks, insurers and other cases where industrial FCF is not the valid headline, V0.1 supports:

```text
SECTOR_DISTRIBUTABLE_CAPITAL
```

The sector-valid distributable-capital framework must be established explicitly.

Industrial Standardized FCF is then not used as the headline cash framework.

## 13. Authority boundary

Owner Cash may validate cash architecture, uncertainty handling and deterministic consistency.

It must never:

- create CASH_ECONOMICS_SCORE;
- determine CASH_ECONOMICS_ELITE;
- invent maintenance capex from depreciation;
- convert non-cash into non-economic by default;
- manufacture a precise Owner Earnings point when maintenance is unresolved;
- override sector-valid cash frameworks;
- write production.

## 14. Module Contract

```text
contracts/orotitan-equity/vnext/modules/
  OWNER_CASH.module-contract.v0.1.json
```

Dependencies:

```text
RETURN_NORMALIZATION
CAPITAL_SEASONING
```

## 15. Runtime implementation

```text
runtime/vnext/modules/owner-cash.ts
```

No model-provider or Azure dependency exists.

## 16. Golden fixtures

```text
tests/fixtures/vnext/owner-cash.v0.1.json
```

Fixture classes:

```text
resolved industrial point Owner Earnings
range with maintenance uncertainty
invalid point with unresolved maintenance
temporary working-capital benefit
SBC double penalty
serial-acquirer cash separation failure
sector-valid distributable capital
deferred-maintenance omission
working-capital double count
```

## 17. Deterministic tests

```text
tests/vnext-owner-cash.test.ts
```

Coverage includes:

```text
Gate 7 Module Contract validation
POINT / RANGE / UNKNOWN discipline
maintenance-resolution firewall
maintenance double-counting firewall
working-capital double-counting firewall
temporary working-capital classification
SBC one-cost-one-penalty rule
acquisition / organic cash separation
deferred-maintenance treatment
per-share diluted-share requirement
sector-valid substitute framework
no score / elite authority
fail-closed finalization
```

## 18. Gate 15 relationship

```text
P0-1 Weak Link Taxonomy              PASS / MERGED
P0-2 Decision State Architecture     PASS / MERGED
P0-3 Return Normalization            PASS / MERGED
P0-4 Capital Seasoning               PASS / MERGED
P0-5 Owner Cash                      CANDIDATE
P0-6 Financing Consistency           NOT STARTED
P0-7 Valuation Assumption Integrity  NOT STARTED
P0-8 Valuation Diagnostic Integrity  NOT STARTED
```

## 19. Acceptance matrix

```text
P0-5-01 cash architecture preserved                         PASS
P0-5-02 reported / standardized / owner cash separated      PASS
P0-5-03 POINT / RANGE / UNKNOWN preserved                   PASS
P0-5-04 false maintenance precision blocked                 PASS
P0-5-05 D&A not treated as maintenance capex                PASS
P0-5-06 maintenance double counting blocked                 PASS
P0-5-07 working-capital double counting blocked             PASS
P0-5-08 temporary WC not structural cash                    PASS
P0-5-09 deferred maintenance reflected                      PASS
P0-5-10 recurring economic costs retained                   PASS
P0-5-11 SBC one-cost-one-penalty                            PASS
P0-5-12 acquisitions outside Standardized FCF by default    PASS
P0-5-13 organic vs acquisition cash separated               PASS
P0-5-14 per-share diluted-share discipline                  PASS
P0-5-15 sector distributable-capital substitute             PASS
P0-5-16 no score / elite authority                          PASS
P0-5-17 no provider/Azure dependency                        PASS
P0-5-18 isolated golden fixtures                            PASS
P0-5-19 deterministic CI                                    PENDING
```

## 20. Current state

```text
P0 MODULE = OWNER_CASH
STATUS    = CANDIDATE
LIVE MODEL REQUIRED = NO
SHADOW RUN MUTATION = FORBIDDEN UNTIL GATE 15
```
