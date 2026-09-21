# OROTITAN_DCF_TIMING_AUTHORITY_V1 — FREEZE V1.0

**Project:** OroTitan Equity Research  
**Authority type:** Global valuation-methodology timing authority  
**Status:** FROZEN — V1.0  
**Freeze date:** 2026-09-21  
**Scope:** DCF timing only  
**Methodology repair classification:** BLOCKING DETERMINISTIC CANONICAL-OUTPUT DEFECT  
**Company calibration:** FORBIDDEN  
**Tesla-specific analytical content:** NONE

## 0. Authority boundary

This authority closes one previously unresolved global valuation convention: the mapping from dated forecast cash flows to discount exponents and present value.

It does not change the existing frozen economic basis:

```text
FCFF -> WACC
FCFE / OWNER EARNINGS -> COST_OF_EQUITY
EXPLICIT PERIOD -> FADE -> STABLE ECONOMICS
TERMINAL VALUE -> frozen terminal framework
POINT-IN-TIME -> preserved
```

It adds no company-specific assumption, score, threshold, hurdle, terminal-growth target, discount-rate target, or valuation conclusion.

Historical outputs are not calibration targets.

```text
CHOOSE TIMING TO REPRODUCE AN OLD COMPANY VALUE = FORBIDDEN
```

## 1. Canonical timing tuple

Every DCF execution must resolve exactly:

```text
TIME_ORIGIN
VALUATION_DATE
REFERENCE_PRICE_DATE
CASH_FLOW_DATES[]
YEAR_FRACTION_CONVENTION
DISCOUNT_EXPONENTS[]
STUB_STATUS
CASH_FLOW_BASIS
DISCOUNT_RATE_BASIS
TERMINAL_VALUE_DATE
EV_TO_EQUITY_BRIDGE_DATE where applicable
ECONOMIC_SHARE_COUNT_DATE
```

Any unresolved required member fails closed.

## 2. TIME_ORIGIN and VALUATION_DATE

```text
TIME_ORIGIN = VALUATION_DATE = DATA_CUTOFF
```

The DCF present-value clock starts at 00:00 UTC on the civil `DATA_CUTOFF` date solely to obtain deterministic whole-day differences. This UTC normalization is computational only; it does not convert the issuer's economic timezone.

`CALCULATION_DATE` is execution metadata and never changes discount exponents.

A rerun on a later calculation date with the same frozen inputs and same `DATA_CUTOFF` must return the same DCF.

Fail states:

```text
DCF_TIME_ORIGIN_MISMATCH
VALUATION_DATE_DATA_CUTOFF_MISMATCH
```

## 3. REFERENCE_PRICE_DATE

`REFERENCE_PRICE_DATE` remains distinct from the DCF time origin.

Rules:

```text
REFERENCE_PRICE_DATE <= VALUATION_DATE
numeric REFERENCE_PRICE -> valid REFERENCE_PRICE_DATE required
REFERENCE_PRICE_DATE does not alter DCF discount exponents
REFERENCE_PRICE_DATE does not move VALUATION_DATE
```

A non-trading-day cutoff may therefore use the last admitted market observation before the cutoff while the DCF remains valued at the cutoff.

This authority creates no new staleness threshold. Existing evidence, valuation-reliability and point-in-time rules determine whether an older reference price is admissible.

Fail states:

```text
REFERENCE_PRICE_DATE_MISSING
REFERENCE_PRICE_AFTER_VALUATION_DATE
REFERENCE_PRICE_STALENESS_UNRESOLVED
```

## 4. CASH_FLOW_DATES and period timing

Canonical DCF payment timing is:

```text
PERIOD_TIMING = END_OF_PERIOD
CASH_FLOW_DATE_i = exact forecast fiscal-period end date_i
```

Beginning-of-period and midpoint discounting are not permitted in V1.0.

Cash-flow dates must be strictly increasing and strictly later than `VALUATION_DATE`.

The issuer's actual fiscal calendar governs. Calendar 31 December must not be substituted for a non-calendar fiscal year. For 52/53-week calendars or irregular fiscal periods, exact forecast period-end dates are required.

Fail states:

```text
UNSUPPORTED_PERIOD_TIMING
CASH_FLOW_DATE_NOT_AFTER_VALUATION_DATE
CASH_FLOW_DATES_NOT_STRICTLY_INCREASING
FISCAL_PERIOD_DATE_UNRESOLVED
FISCAL_YEAR_MISMATCH_UNRESOLVED
```

## 5. YEAR_FRACTIONS and DISCOUNT_EXPONENTS

Canonical day-count convention:

```text
YEAR_FRACTION_CONVENTION = ACT/365F
t_i = calendar_days(VALUATION_DATE, CASH_FLOW_DATE_i) / 365
PV_i = CF_i / (1 + r)^t_i
```

Leap years use the actual numerator and the fixed denominator 365. A 366-day interval therefore has exponent `366/365`.

No 30/360, ACT/ACT, whole-year integer exponent, midpoint exponent, or analyst-selected exponent is allowed.

No intermediate rounding is allowed. Presentation may round only after the full calculation.

Fail states:

```text
UNSUPPORTED_YEAR_FRACTION_CONVENTION
NONFINITE_DISCOUNT_EXPONENT
INTERMEDIATE_ROUNDING_AS_INPUT
```

## 6. STUB_PERIODS

The first forecast interval always begins at `VALUATION_DATE`.

If the next forecast fiscal-period end is not one full modeled period after `VALUATION_DATE`, the first interval is a stub.

Canonical rule:

```text
FIRST_FORECAST_PERIOD_START = VALUATION_DATE
FIRST_FORECAST_CASH_FLOW = cash flow economically attributable only to the post-origin stub
AUTOMATIC PRORATION OF A FULL-YEAR CASH FLOW = FORBIDDEN
SILENT OMISSION OF THE STUB = FORBIDDEN
```

The valuation model must support the stub from actual subannual evidence or an explicit forecast construction. If it cannot, the DCF is not deterministically executable.

Subsequent forecast intervals must begin on the preceding forecast payment date.

Fail states:

```text
STUB_CASH_FLOW_UNSUPPORTED
STUB_FULL_YEAR_AUTOPRORATION_FORBIDDEN
FORECAST_PERIOD_CHAIN_BROKEN
```

## 7. FCFF

For enterprise DCF:

```text
CASH_FLOW_BASIS = FCFF
DISCOUNT_RATE_BASIS = WACC
DCF_PRESENT_VALUE = ENTERPRISE_VALUE at VALUATION_DATE
```

Any FCFF/cost-of-equity mismatch or mixed-basis stream fails closed.

Fail states:

```text
FCFF_DISCOUNT_RATE_BASIS_MISMATCH
MIXED_CASH_FLOW_BASIS
```

## 8. FCFE

For direct equity DCF:

```text
CASH_FLOW_BASIS = FCFE
DISCOUNT_RATE_BASIS = COST_OF_EQUITY
DCF_PRESENT_VALUE = EQUITY_VALUE at VALUATION_DATE
```

The ordinary FCFF enterprise-to-equity bridge must not be applied a second time.

Fail states:

```text
FCFE_DISCOUNT_RATE_BASIS_MISMATCH
EQUITY_DCF_EV_BRIDGE_FORBIDDEN
```

## 9. OWNER_EARNINGS

When Owner Earnings are used as an equity cash-flow valuation basis:

```text
CASH_FLOW_BASIS = OWNER_EARNINGS
DISCOUNT_RATE_BASIS = COST_OF_EQUITY
DCF_PRESENT_VALUE = EQUITY_VALUE at VALUATION_DATE
```

If Owner Earnings cannot be supportably bounded under the frozen forensic method, timing determinism does not make the valuation assessable.

Fail states:

```text
OWNER_EARNINGS_DISCOUNT_RATE_BASIS_MISMATCH
OWNER_EARNINGS_BASIS_UNSUPPORTED
EQUITY_DCF_EV_BRIDGE_FORBIDDEN
```

## 10. TERMINAL_VALUE_TIMING

The terminal value is valued at the end of the final explicit/fade forecast period.

```text
TERMINAL_VALUE_DATE = CASH_FLOW_DATE_N
TV_N = CF_(N+1) / (r - g) where the frozen perpetuity framework is applicable
CF_(N+1) = terminal-basis cash flow one period after CF_N under stable economics
PV(TV) = TV_N / (1 + r)^t_N
t_N = ACT/365F(VALUATION_DATE, TERMINAL_VALUE_DATE)
```

The terminal value and the final explicit cash flow therefore share the same discount exponent.

No extra one-year discount, midpoint adjustment, beginning-of-period adjustment, or exit-multiple timing substitution is permitted.

The terminal cash-flow basis must match the DCF basis and `r > g` must hold for a perpetuity-growth terminal value.

Fail states:

```text
TERMINAL_VALUE_DATE_MISMATCH
TERMINAL_BASIS_MISMATCH
TERMINAL_G_NOT_LESS_THAN_R
TERMINAL_STABLE_ECONOMICS_UNRESOLVED
```

## 11. EV_TO_EQUITY_BRIDGE

Only an FCFF enterprise DCF uses the ordinary EV-to-equity bridge.

```text
BRIDGE_DATE = VALUATION_DATE
EQUITY_VALUE_t0 = ENTERPRISE_VALUE_t0 + NET_EQUITY_BRIDGE_ADJUSTMENT_t0
```

The bridge adjustment must reflect the frozen EV-bridge definitions for debt, lease debt where applicable, preferred claims, NCI, pension/debt-like items, contingent consideration, excess cash and non-operating investments.

Bridge items are point-in-time claims/assets. They are not discounted as forecast cash flows.

If the latest reported balance-sheet date differs from the valuation date, the analyst must explicitly reconcile/roll the material bridge to the valuation date using only information available by `DATA_CUTOFF`. An unreconciled material date mismatch blocks the DCF.

Fail states:

```text
EV_BRIDGE_REQUIRED_FOR_FCFF
EV_BRIDGE_DATE_MISMATCH
EV_BRIDGE_ROLLFORWARD_UNRESOLVED
EQUITY_DCF_EV_BRIDGE_FORBIDDEN
```

## 12. PER_SHARE_TIMING

Per-share intrinsic value uses the economic share count at the valuation date.

```text
SHARE_COUNT_DATE = VALUATION_DATE
INTRINSIC_VALUE_PER_SHARE = EQUITY_VALUE_t0 / ECONOMIC_SHARE_COUNT_t0
```

Weighted-average diluted shares used for EPS are not automatically a valid point-in-time denominator.

Options, RSUs, convertibles, issuance, buybacks and other dilution must be handled consistently with the frozen share-count method. A future dilution effect already modeled in cash flows or claims must not be double counted in the denominator.

Fail states:

```text
PER_SHARE_DATE_MISMATCH
ECONOMIC_SHARE_COUNT_UNRESOLVED
NONPOSITIVE_ECONOMIC_SHARE_COUNT
DILUTION_DOUBLE_COUNT_UNRESOLVED
```

## 13. Deterministic reference vectors

All values below are synthetic controls. They are not company evidence and must not be calibrated to any historical valuation.

### R1 — FCFF, stub, EV bridge

```text
VALUATION_DATE = 2026-09-19
CF1 = 100.0 at 2026-12-31
CF2 = 120.0 at 2027-12-31
WACC = 0.10
TERMINAL_G = 0.03
NET_EQUITY_BRIDGE_ADJUSTMENT = -150.0
ECONOMIC_SHARE_COUNT = 10.0

t1 = 0.2821917808219178
t2 = 1.2821917808219179
PV_CF1 = 97.3462720337052
PV_CF2 = 106.19593312767839
TV = 1765.7142857142858
PV_TV = 1562.5973017358392
ENTERPRISE_VALUE = 1766.1395068972229
EQUITY_VALUE = 1616.1395068972229
PER_SHARE_VALUE = 161.6139506897223
```

### R2 — FCFE, irregular first stub

```text
VALUATION_DATE = 2027-01-15
CF1 = 40.0 at 2027-06-30
CF2 = 55.0 at 2028-06-30
COST_OF_EQUITY = 0.12
TERMINAL_G = 0.02
ECONOMIC_SHARE_COUNT = 5.0

t1 = 0.4547945205479452
t2 = 1.4575342465753425
PV_CF1 = 37.99057828118631
PV_CF2 = 46.62573981939283
TV = 561.0000000000001
PV_TV = 475.582546157807
EQUITY_VALUE = 560.1988642583862
PER_SHARE_VALUE = 112.03977285167723
```

### R3 — Owner Earnings, leap-year numerator

```text
VALUATION_DATE = 2027-12-31
CF1 = 30.0 at 2028-12-31
COST_OF_EQUITY = 0.09
TERMINAL_G = 0.025

t1 = 1.0027397260273974
PV_CF1 = 27.5164382915453
TV = 473.076923076923
PV_TV = 433.91306536667577
EQUITY_VALUE = 461.42950365822105
```

Regression comparisons use a numeric tolerance of `1e-12 * max(1, abs(expected))`; no fixture may be changed to match a company result.

## 14. Adversarial acceptance matrix

The frozen implementation must prove at minimum:

```text
T01 valuation date != data cutoff -> FAIL
T02 reference price date after valuation date -> FAIL
T03 cash-flow date at/before origin -> FAIL
T04 non-increasing payment dates -> FAIL
T05 broken forecast period chain -> FAIL
T06 unsupported/autoprorated stub -> FAIL
T07 FCFF + cost of equity -> FAIL
T08 FCFE + WACC -> FAIL
T09 Owner Earnings + WACC -> FAIL
T10 equity DCF + EV bridge -> FAIL
T11 FCFF without valuation-date EV bridge -> FAIL
T12 terminal date != final cash-flow date -> FAIL
T13 g >= r -> FAIL
T14 bridge date != valuation date -> FAIL
T15 share-count date != valuation date -> FAIL
T16 nonpositive share count -> FAIL
T17 unresolved fiscal calendar -> FAIL
T18 historical-output calibration flag -> FAIL
T19 ACT/365F leap interval -> exact 366/365
T20 calculation-date movement -> no output change
T21 R1 exact reference outputs -> PASS
T22 R2 exact reference outputs -> PASS
T23 R3 exact reference outputs -> PASS
T24 two independent implementations -> numerically equivalent
```

Required final state:

```text
REGRESSION = PASS
INDEPENDENT_IMPLEMENTATION_EQUIVALENCE = PASS
UNRESOLVED_POLICY_BRANCHES = 0
```

## 15. Contract dependency resolution

This authority is an additive higher-order valuation convention for the previously unspecified timing dimension.

Existing authorities remain unchanged in meaning:

```text
V3 analysis standard -> economic DCF framework, economic share-count authority and basis matching
V3 execution process -> stage sequencing, checkpoint semantics, point-in-time and immutable run pins
V3 pilotage -> exact run routing and historical Contract Set firewall
V3 Deep Dive -> Valuation consumes frozen valuation policy/conventions and exact upstream lock
V3 integration -> projects certified outputs, never re-values
I2 / I3-B -> unchanged except that they consume the successor run's governed valuation outputs
```

This authority is a blocking deterministic canonical-output repair. It does not reopen business-quality, scoring, investment-policy, economic-share-count, Certification or terminal-gate methodology.

A run created before this authority was pinned must never be rebound in place.

## 16. Controlled same-cutoff methodology replay

A controlled successor run is permitted only to repair a historical run whose valuation output depends on the timing convention closed by this authority.

`SUCCESSOR` is lineage, not a legal `RUN_TYPE`.

For an unpublished parent with no current canonical snapshot:

```text
LEGAL_RUN_TYPE = INITIAL
PARENT_RUN_ID = exact predecessor RUN_ID
BASELINE_SNAPSHOT_ID = NULL
DATA_CUTOFF = exact parent DATA_CUTOFF
CONTRACT_SET = active successor Contract Set containing this authority
FIRST_REGISTRY_STAGE = RESEARCH
FIRST_ANALYTICAL_PHASE = VALUATION, but only after exact upstream non-valuation inputs are revalidated
```

For a parent with a valid current canonical snapshot, ordinary frozen REFRESH routing governs instead.

### 16.1 Transactional parent CAS

Planning-time observation is insufficient. Successor creation must transactionally lock and revalidate the parent row before inserting the child.

The successor creation transaction must require exact caller-supplied expectations for:

```text
PARENT_RUN_ID
PARENT_EXPECTED_STATE_VERSION
PARENT_EXPECTED_RUN_STATUS = ACTIVE
PARENT_EXPECTED_CURRENT_STAGE = DEEP_DIVE
PARENT_EXPECTED_DATA_CUTOFF
PARENT_EXPECTED_CONTRACT_SET_SHA256
PARENT_EXPECTED_ISSUER_ID
PARENT_EXPECTED_SECURITY_ID
PARENT_EXPECTED_DOSSIER_ID
PARENT_PUBLISHED_AT = NULL
PARENT_CANCELLED_AT = NULL
CURRENT_CANONICAL_SNAPSHOT_ID = NULL
```

Any mismatch fails closed before child insertion.

Required fail states include:

```text
SUCCESSOR_PARENT_NOT_FOUND
SUCCESSOR_PARENT_STATE_VERSION_MISMATCH
SUCCESSOR_PARENT_STATUS_MISMATCH
SUCCESSOR_PARENT_STAGE_MISMATCH
SUCCESSOR_PARENT_CUTOFF_MISMATCH
SUCCESSOR_PARENT_CONTRACT_SET_MISMATCH
SUCCESSOR_PARENT_IDENTITY_MISMATCH
SUCCESSOR_PARENT_TERMINAL
SUCCESSOR_BASELINE_MISMATCH
SUCCESSOR_PARENT_ALREADY_ON_ACTIVE_CONTRACT_SET
```

### 16.2 Cross-run artifact authority

Same-cutoff replay does not permit conversational reconstruction.

Only exact, persisted, hash-verified non-valuation artifacts may cross the run boundary, and every crossing must be represented by explicit `REVALIDATES` and/or `DERIVED_FROM` lineage edges.

The timing repair is valuation-only. Therefore an exact parent `FUNDAMENTALS_LOCK` may be revalidated without changing its analytical content when all of the following hold:

```text
same issuer/security/dossier
same DATA_CUTOFF
artifact bytes/hash resolve exactly
its complete producer lineage resolves
the timing repair has no upstream fundamental-method dependency
no post-cutoff evidence is introduced
no fundamental judgment changes
no unresolved conflict requires reopening Fundamentals
```

The successor Research stage exists only to perform and persist this same-cutoff revalidation boundary. It must not perform an information refresh.

After Research finalization, the successor Deep Dive may create a revalidation checkpoint that binds the exact revalidated non-valuation lock. No new fundamental judgment is created. Valuation is then the first analytical phase.

The following parent artifacts may not be revalidated across this methodology boundary:

```text
VALUATION_FULL_PRECISION_INPUTS
VALUATION_ARTIFACT
VALUATION_LOCK
valuation-derived Calculation Ledger content
valuation-derived expected-return / price-ladder / reverse-DCF outputs
Certification outputs that depend on the old valuation
```

They remain historical audit records only.

If the parent non-valuation artifacts cannot be revalidated without analytical change, the successor must route to the earliest affected upstream phase instead of forcing a Valuation-only replay.

### 16.3 Replay invariants

```text
PARENT RUN = immutable
PARENT CONTRACT SET = immutable
PARENT DATA_CUTOFF = immutable
PARENT VALUATION OUTPUTS = historical only
NO NEW POST-CUTOFF EVIDENCE
NO NEW FUNDAMENTAL JUDGMENT
NO CALIBRATION TO PARENT VALUATION
NO OLD VALUATION_LOCK REUSE
NO NEW VALUATION_LOCK IN THE GOVERNANCE-REPAIR WORKSTREAM
CERTIFICATION / INTEGRATION remain downstream and are not executed by this governance repair
```

## 17. Parent firewall

A methodology repair must not mutate an existing run's immutable fields:

```text
run_id
entry_path
canonical_mode
run_type
data_cutoff
process_version
pilotage_contract_version
contract_pins
contract_set_sha256
```

No historical Stage Manifest, artifact, edge, event, snapshot or canonical pointer is deleted or rewritten. No historical Valuation Lock is deleted or rewritten.

The governance-repair workstream may freeze methodology, activate a successor Contract Set, and deploy the transactional successor-creation control. It must not create a Tesla successor, execute Tesla Valuation, create a new Tesla Valuation Lock, certify Tesla, integrate Tesla or publish Tesla.

## 18. Status

```text
GLOBAL_DCF_TIMING_AUTHORITY = FROZEN
VERSION = 1.0
UNRESOLVED_POLICY_BRANCHES = 0
COMPANY_CALIBRATION = FORBIDDEN
```
