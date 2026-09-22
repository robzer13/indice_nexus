# OROTITAN_VNEXT_P0_RETURN_NORMALIZATION_V0.1

**Project:** OroTitan Equity Research  
**Status:** CANDIDATE — GATE 15 / P0-3  
**Methodology change:** NO  
**Provider dependency:** NONE  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

This module implements the third Gate 15 P0 analytical component: Return Normalization.

It operationalizes the frozen Return on Capital / Marginal Return architecture without changing formulas, thresholds, weights, or Return Quality scoring.

Core invariants consumed by this module:

```text
BUSINESS ROIC
!= CAPITAL-ALLOCATION ROIC
!= MARGINAL ROIC

HIGH HISTORICAL ROIC
!= HIGH FUTURE ROIIC

CALCULABLE
!= ECONOMICALLY INTERPRETABLE
```

## 1. Frozen return architecture

For ordinary non-financial businesses:

```text
STANDARD ROIC
ALL-IN ROIC      when acquisition capital is material
ROIIC            when economically interpretable
```

Diagnostics:

```text
ROIC EX-GOODWILL
R&D-ADJUSTED ROIC when material
```

Special:

```text
ACQUISITION / COHORT RETURN
```

Valid outcome:

```text
ROIC NOT ECONOMICALLY INTERPRETABLE
```

Sector-valid substitutes are explicitly supported when industrial ROIC is not applicable.

## 2. Standard ROIC

The module does not redefine the frozen calculation:

```text
STANDARD_ROIC
=
NORMALIZED_NOPAT
/
AVERAGE_OPERATING_INVESTED_CAPITAL
```

and:

```text
NORMALIZED_NOPAT
=
NORMALIZED_OPERATING_EBIT
*
(1 - NORMALIZED_OPERATING_TAX_RATE)
```

Arithmetic belongs in the Calculation Ledger / deterministic calculation layer. This module validates whether the selected return framework is economically usable.

## 3. Acquisition capital

When acquisition capital is material:

```text
ALL_IN_ROIC = REQUIRED headline/core measure
ACQUISITION_COHORT_RETURN = REQUIRED special measure
ROIC_EX_GOODWILL = diagnostic only
```

The module rejects any architecture that treats ex-goodwill return as the sole economic headline for a material acquirer.

Historical acquisition capital destruction cannot disappear merely because accounting impairment reduced book goodwill.

## 4. Negative / near-zero invested capital

When invested capital is near zero or negative and produces explosive ratios:

```text
ROIC = NOT_INTERPRETABLE
```

The module therefore:

- rejects denominator-driven Standard / All-In ROIC as headline quality evidence;
- requires traceable alternative economics;
- preserves ROIC as a diagnostic if useful;
- never awards a quality bonus for a numerically extreme denominator artifact.

Alternative economics may include:

```text
NOPAT margin
incremental margin
capital turnover
return on incremental operating assets
unit economics
return on discretionary growth spend
cohort economics
```

## 5. ROIIC interpretability gate

Frozen formula:

```text
ROIIC
=
delta NORMALIZED NOPAT
/
delta ECONOMIC INVESTED CAPITAL
```

Interpretability requires:

```text
delta IC > 0
delta IC material
same perimeter
no unexplained major M&A
no major divestiture
no major accounting reclassification
investment and profit periods linked
NOPAT normalized
```

Deterministic outcome:

```text
all gate conditions satisfied
-> INTERPRETABLE

any explicit gate failure
-> NOT_INTERPRETABLE

material gate uncertainty
-> UNKNOWN
```

Near-zero or negative delta IC is never converted into an unstable extreme ROIIC.

## 6. ROIIC windows

Frozen windows:

```text
1Y = DIAGNOSTIC ONLY
3Y = DEFAULT PRIMARY
5Y = CORROBORATION / LAG-AWARE
```

V0.1 therefore blocks a 1-year window from being used as the primary ROIIC normalization.

## 7. R&D-adjusted ROIC

R&D-adjusted ROIC remains diagnostic and is permitted only when R&D is materially recurring/multi-period and expensing distorts economic comparison.

Frozen symmetry requirement:

```text
current R&D add-back
requires
unamortized R&D asset in denominator
+
R&D amortization
```

The module fails closed when a current-period R&D add-back is used without denominator and amortization symmetry.

## 8. Interpretation metadata

The module always preserves separately:

```text
DATA_QUALITY
ATTRIBUTABILITY
INTERPRETABILITY
```

UNKNOWN is preserved explicitly and is never coerced to a favorable state.

## 9. Authority boundary

The module may validate framework consistency and deterministic interpretability gates.

It must never:

- create a Return Quality score;
- determine RETURN_QUALITY_ELITE;
- create new ROIC/ROIIC formulas;
- invent economic thresholds;
- infer future ROIIC from historical ROIC;
- turn a calculable ratio into an economically interpretable one by default;
- write production.

## 10. Module Contract

```text
contracts/orotitan-equity/vnext/modules/
  RETURN_NORMALIZATION.module-contract.v0.1.json
```

## 11. Runtime implementation

```text
runtime/vnext/modules/return-normalization.ts
```

No model-provider or Azure dependency exists.

## 12. Golden fixtures

```text
tests/fixtures/vnext/return-normalization.v0.1.json
```

Fixture classes:

```text
ordinary industrial + interpretable 3Y ROIIC
serial acquirer + All-In ROIC + cohort return
near-zero invested capital + alternative economics
near-zero delta IC + invalid ROIIC headline
R&D adjustment symmetry failure
sector-valid substitute framework
```

## 13. Deterministic tests

```text
tests/vnext-return-normalization.test.ts
```

Coverage includes:

```text
Gate 7 Module Contract validation
ROIIC interpretability gate
1Y diagnostic-only guard
All-In ROIC requirement
acquisition cohort-return requirement
ex-goodwill diagnostic-only rule
near-zero denominator firewall
alternative economics requirement
R&D numerator/denominator symmetry
sector-valid substitute support
UNKNOWN metadata preservation
no score / elite-state authority
fail-closed finalization
```

## 14. Gate 15 relationship

```text
P0-1 Weak Link Taxonomy              PASS / MERGED
P0-2 Decision State Architecture     PASS / MERGED
P0-3 Return Normalization            CANDIDATE
P0-4 Capital Seasoning               NOT STARTED
P0-5 Owner Cash                      NOT STARTED
P0-6 Financing Consistency           NOT STARTED
P0-7 Valuation Assumption Integrity  NOT STARTED
P0-8 Valuation Diagnostic Integrity  NOT STARTED
```

## 15. Acceptance matrix

```text
P0-3-01 frozen return architecture preserved                 PASS
P0-3-02 Standard ROIC headline rule preserved                PASS
P0-3-03 All-In ROIC for material acquisition capital         PASS
P0-3-04 ex-goodwill remains diagnostic                       PASS
P0-3-05 acquisition cohort return requirement                PASS
P0-3-06 near-zero denominator firewall                       PASS
P0-3-07 alternative economics required when ROIC unusable    PASS
P0-3-08 frozen ROIIC interpretability gate                   PASS
P0-3-09 1Y ROIIC diagnostic-only rule                        PASS
P0-3-10 R&D adjustment symmetry                              PASS
P0-3-11 sector-valid framework support                       PASS
P0-3-12 data quality / attribution / interpretability split  PASS
P0-3-13 UNKNOWN preserved                                    PASS
P0-3-14 no score / elite authority                           PASS
P0-3-15 no provider/Azure dependency                         PASS
P0-3-16 isolated golden fixtures                             PASS
P0-3-17 deterministic CI                                     PENDING
```

## 16. Current state

```text
P0 MODULE = RETURN_NORMALIZATION
STATUS    = CANDIDATE
LIVE MODEL REQUIRED = NO
SHADOW RUN MUTATION = FORBIDDEN UNTIL GATE 15
```
