# OROTITAN_VNEXT_P0_FINANCING_CONSISTENCY_V0.1

**Project:** OroTitan Equity Research  
**Status:** CANDIDATE — GATE 15 / P0-6  
**Methodology change:** NO  
**Provider dependency:** NONE  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

This module implements the sixth Gate 15 P0 analytical component: Financing Consistency.

It checks whether operating cash, debt, equity, leases, supplier finance, factoring, distributions, buybacks, acquisitions and asset sales are reconciled consistently.

It does not create a leverage score and does not replace Capital Allocation or Resilience judgment.

## 1. Canonical financing bridge

The frozen multi-year capital-allocation cash bridge is:

```text
FCF GENERATED
+ DEBT RAISED
+ EQUITY RAISED
+ ASSET SALES
↓
CAPEX
M&A
DIVIDENDS
BUYBACKS
DEBT REPAYMENT
CASH ACCUMULATION
```

V0.1 requires this sources/uses bridge and the net-cash/debt bridge to reconcile before financing consistency can finalize.

## 2. Debt

When debt funding is material, V0.1 requires consistent treatment across:

```text
debt cash flows
cash interest
net cash / debt
enterprise value
```

No fixed leverage threshold or debt-capacity formula is introduced.

## 3. Equity issuance

Material equity financing must be reflected in economic shares.

This preserves the frozen principle that per-share economics require the actual economic share count rather than ignoring financing dilution.

## 4. Supplier finance

When supplier-finance balances have financing substance, debt-like treatment must be reconciled consistently.

V0.1 therefore rejects a material supplier-finance exposure that remains treated purely as operating payables while financing substance is established.

## 5. Factoring / receivable sales

Material factoring, receivable sales, securitization or invoice discounting require explicit reconciliation of any financing transfer.

```text
CFO IMPROVEMENT
!=
IMPROVED OPERATING ECONOMICS
```

when the improvement is produced by financing transfer.

## 6. Leases

Material lease treatment must remain symmetric across:

```text
cash-flow presentation
ROIC treatment
EV treatment
```

V0.1 rejects inconsistent cross-block lease treatment.

## 7. Acquisition funding

When acquisition funding is material, the funding bridge must resolve the actual mix and consequences of cash, debt and equity.

This module validates financing consistency only. Acquisition return remains Return Quality / cohort economics, while allocation quality remains Capital Allocation.

## 8. Buybacks

Frozen rule:

```text
do not assume buybacks beyond economically available cash
without a financing bridge
```

If buybacks are assumed or executed, V0.1 requires:

```text
funding bridge established
+
net economic share-count effect reconciled
```

Gross repurchases are not automatically shareholder yield.

## 9. Dividends

Dividends are cash distributions but must remain consistent with retained-capital and reinvestment needs.

V0.1 rejects a distribution assumption that conflicts with those needs.

## 10. Asset sales

Material asset-sale proceeds must remain separate from recurring operating cash generation.

Asset sales may finance capital allocation, but they may not be silently relabeled as sustainable owner cash.

## 11. UNKNOWN discipline

Unknown financing materiality remains explicit.

Examples:

```text
SUPPLIER_FINANCE_MATERIALITY_UNKNOWN
FACTORING_MATERIALITY_UNKNOWN
LEASE_MATERIALITY_UNKNOWN
ACQUISITION_FUNDING_MATERIALITY_UNKNOWN
```

UNKNOWN is a limitation, not an automatic favorable financing conclusion.

## 12. Authority boundary

The module must never:

- create a leverage score;
- create or modify Capital Allocation score;
- create or modify Resilience score;
- invent debt-capacity thresholds;
- judge acquisition returns;
- infer that debt or equity use is good/bad merely from activity;
- write production.

## 13. Module Contract

```text
contracts/orotitan-equity/vnext/modules/
  FINANCING_CONSISTENCY.module-contract.v0.1.json
```

Dependency:

```text
OWNER_CASH
```

## 14. Runtime implementation

```text
runtime/vnext/modules/financing-consistency.ts
```

No model-provider or Azure dependency exists.

## 15. Golden fixtures

```text
tests/fixtures/vnext/financing-consistency.v0.1.json
```

Fixture classes:

```text
clean financing bridge
material supplier finance
material factoring
lease-treatment asymmetry
buyback without funding bridge
dividend / retained-capital conflict
material acquisition funding unresolved
equity issuance omitted from economic shares
UNKNOWN supplier-finance materiality
asset-sale operating-cash misclassification
capital-allocation cash-bridge failure
```

## 16. Deterministic tests

```text
tests/vnext-financing-consistency.test.ts
```

Coverage includes:

```text
Gate 7 Module Contract validation
capital-allocation cash bridge
net cash/debt reconciliation
supplier-finance debt-like treatment
factoring financing-transfer treatment
lease cash/ROIC/EV symmetry
buyback funding + share-count bridge
dividend retained-capital consistency
acquisition funding bridge
equity issuance / economic shares
asset-sale operating-cash separation
UNKNOWN preservation
no scoring authority
fail-closed finalization
```

## 17. Gate 15 relationship

```text
P0-1 Weak Link Taxonomy              PASS / MERGED
P0-2 Decision State Architecture     PASS / MERGED
P0-3 Return Normalization            PASS / MERGED
P0-4 Capital Seasoning               PASS / MERGED
P0-5 Owner Cash                      PASS / MERGED
P0-6 Financing Consistency           CANDIDATE
P0-7 Valuation Assumption Integrity  NOT STARTED
P0-8 Valuation Diagnostic Integrity  NOT STARTED
```

## 18. Acceptance matrix

```text
P0-6-01 sources/uses financing bridge reconciled              PASS
P0-6-02 net cash/debt bridge reconciled                       PASS
P0-6-03 debt cash/interest/EV consistency                     PASS
P0-6-04 equity issuance reflected in economic shares          PASS
P0-6-05 supplier finance debt-like reconciliation             PASS
P0-6-06 factoring financing-transfer reconciliation           PASS
P0-6-07 lease cash/ROIC/EV consistency                        PASS
P0-6-08 acquisition funding bridge                            PASS
P0-6-09 buyback financing bridge                              PASS
P0-6-10 buyback share-count reconciliation                    PASS
P0-6-11 dividend retained-capital consistency                 PASS
P0-6-12 asset sales separated from operating cash             PASS
P0-6-13 UNKNOWN preserved                                     PASS
P0-6-14 no leverage/capital-allocation/resilience score       PASS
P0-6-15 no provider/Azure dependency                          PASS
P0-6-16 isolated golden fixtures                              PASS
P0-6-17 deterministic CI                                      PENDING
```

## 19. Current state

```text
P0 MODULE = FINANCING_CONSISTENCY
STATUS    = CANDIDATE
LIVE MODEL REQUIRED = NO
SHADOW RUN MUTATION = FORBIDDEN UNTIL GATE 15
```
