# OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1 - FREEZE V1.0

**Project:** OroTitan Equity Research
**Authority type:** Global valuation-methodology date-alignment authority
**Status:** FROZEN - V1.0
**Freeze date:** 2026-09-23
**Governance class:** GLOBAL_VALUATION_METHODOLOGY
**Repair classification:** PINNED_AUTHORITY_CONFLICT_RESOLUTION
**Company calibration:** FORBIDDEN
**Historical run rewrite:** FORBIDDEN

## 0. Problem and authority boundary

The active V3.0.1 Contract Set simultaneously pins:

1. `OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0`, whose Section 3 states:
   - `VALUATION_DATE = REFERENCE_PRICE_DATE`;
   - `SHARE_COUNT_AS_OF_DATE = REFERENCE_PRICE_DATE`.

2. `OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0`, whose Sections 2, 11 and 12 state:
   - `TIME_ORIGIN = VALUATION_DATE = DATA_CUTOFF`;
   - `EV_TO_EQUITY_BRIDGE_DATE = VALUATION_DATE` where applicable;
   - `SHARE_COUNT_DATE = VALUATION_DATE`;
   - `REFERENCE_PRICE_DATE` does not move `VALUATION_DATE`.

When `REFERENCE_PRICE_DATE <> DATA_CUTOFF`, these frozen clauses cannot all be true.

This authority repairs only valuation date alignment. It does not alter the economic definition of shares, bound construction, DCF period timing, forecast economics, discount-rate methodology, scoring, Investment Policy, Certification or terminal gates.

## 1. Canonical date architecture

For every run governed by this authority:

```text
TIME_ORIGIN
= VALUATION_DATE
= DATA_CUTOFF

ECONOMIC_SHARE_COUNT_DATE
= SHARE_COUNT_AS_OF_DATE
= VALUATION_DATE

EV_TO_EQUITY_BRIDGE_DATE
= VALUATION_DATE
when an FCFF enterprise-to-equity bridge is required

PER_SHARE_OUTPUT_DATE
= VALUATION_DATE
```

`REFERENCE_PRICE_DATE` remains a distinct observed-market date:

```text
REFERENCE_PRICE_DATE <= VALUATION_DATE
REFERENCE_PRICE_DATE does not move VALUATION_DATE
REFERENCE_PRICE_DATE does not move TIME_ORIGIN
REFERENCE_PRICE_DATE does not move SHARE_COUNT_AS_OF_DATE
```

A numeric reference price requires a valid admitted `REFERENCE_PRICE_DATE`.

## 2. Market-cap date

An instantaneous observed market capitalization is a same-date market object.

```text
MARKET_CAP_DATE = REFERENCE_PRICE_DATE
MARKET_CAP = REFERENCE_PRICE(REFERENCE_PRICE_DATE)
           * ECONOMIC_SHARE_COUNT(REFERENCE_PRICE_DATE)
```

If `REFERENCE_PRICE_DATE < VALUATION_DATE`, the market-cap denominator is a separate evaluation of the same economic-share-count function at `REFERENCE_PRICE_DATE`. The valuation-date denominator must not be transported backward, and the reference-date denominator must not be transported forward.

A market capitalization measured at `REFERENCE_PRICE_DATE` must not be labelled current at `VALUATION_DATE`.

If a same-date economic share count for the market-cap observation cannot be demonstrated or rigorously bounded, `MARKET_CAP = NOT_ASSESSABLE`. This does not by itself make a direct per-share intrinsic valuation unassessable if its valuation-date denominator is independently governed.

## 3. Non-trading-day cutoff

When `DATA_CUTOFF` is a non-trading day, the last admitted market observation before the cutoff may remain the `REFERENCE_PRICE` under the existing DCF-timing and staleness rules.

Therefore:

```text
VALUATION_DATE = non-trading DATA_CUTOFF
REFERENCE_PRICE_DATE = last admitted prior trading observation
MARKET_CAP_DATE = REFERENCE_PRICE_DATE
PER_SHARE_OUTPUT_DATE = VALUATION_DATE
```

No synthetic price is created for the non-trading day.

## 4. Economic share-count closure at VALUATION_DATE

The representation order and all hard-bound rules of `OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0` remain unchanged:

```text
EXACT_SCALAR
-> else RIGOROUS_BOUND
-> else UNKNOWN
```

The target date is changed only for valuation-denominator use:

```text
BOUND_EFFECTIVE_DATE = SHARE_COUNT_AS_OF_DATE = VALUATION_DATE
```

A denominator, exact count or bound established at any earlier date is not current merely because no later movement is known.

```text
SILENCE = ZERO MOVEMENT
```

is forbidden.

To move an earlier exact count or bound to `VALUATION_DATE`, every relevant denominator-changing movement class over the intervening interval must be exact, proven not applicable/zero, or hard-bounded under the frozen share-count method while preserving state conservation and correlation.

If this cannot be demonstrated:

```text
ECONOMIC_SHARE_COUNT = UNKNOWN
DENOMINATOR_VALUATION_ADMISSION = NO
DENOMINATOR_DEPENDENT_OUTPUTS = NOT_ASSESSABLE
```

No midpoint, endpoint, weighted-average EPS shares, stale exact value or vendor estimate may substitute.

## 5. EV-to-equity bridge closure

For FCFF:

```text
EV_TO_EQUITY_BRIDGE_DATE = VALUATION_DATE
```

A balance-sheet bridge from an earlier reported date may be used only after explicit cutoff-compliant reconciliation or roll-forward of every material bridge item required by the frozen EV definition.

If a material bridge item cannot be closed or bounded to `VALUATION_DATE`:

```text
EV_TO_EQUITY_BRIDGE = UNRESOLVED
ENTERPRISE_VALUE_CURRENT = NOT_ASSESSABLE
FCFF_EQUITY_VALUE = NOT_ASSESSABLE
```

The absence of disclosed movement is not evidence of zero movement.

For FCFE or supported Owner Earnings equity DCF, the ordinary FCFF EV bridge remains forbidden and `EV_TO_EQUITY_BRIDGE_DATE = NOT_APPLICABLE`.

## 6. Per-share outputs

Every valuation-date per-share output uses the governed valuation-date denominator:

```text
PER_SHARE_OUTPUT_DATE = VALUATION_DATE
INTRINSIC_VALUE_PER_SHARE
= EQUITY_VALUE(VALUATION_DATE)
  / ECONOMIC_SHARE_COUNT(VALUATION_DATE)
```

For a lawful share-count bound `S in [S_LOW,S_HIGH]` and scalar positive equity value `E`:

```text
INTRINSIC_VALUE_PER_SHARE
= [E / S_HIGH ; E / S_LOW]
```

When multiple uncertain variables are causally linked, optimize over the joint feasible set. A Cartesian product of independent marginal bounds is forbidden unless independence is itself established.

## 7. Reference-price dependent outputs

A reference price may be used for expected-return, MOS, reverse-DCF or market-expectation comparison only when its own date and staleness are admitted under the frozen point-in-time rules.

The temporal gap is explicit:

```text
REFERENCE_PRICE_DATE < VALUATION_DATE
```

does not make the price a future observation and does not change the valuation date.

If the price is not admitted, price-dependent outputs are `NOT_ASSESSABLE`; intrinsic-value outputs that do not require price may remain independently assessable.

## 8. Categorical outputs

For any range or joint feasible set, a categorical output is emitted only if the frozen category is identical for every admissible state.

Otherwise use the existing frozen `NOT_ASSESSABLE` or `NOT_AVAILABLE` semantics as applicable. No category is selected from a midpoint or preferred endpoint.

## 9. Explicit fail states

The following are hard fail states for affected outputs:

```text
VALUATION_DATE_DATA_CUTOFF_MISMATCH
TIME_ORIGIN_MISMATCH
REFERENCE_PRICE_AFTER_VALUATION_DATE
REFERENCE_PRICE_STALENESS_UNRESOLVED
ECONOMIC_SHARE_COUNT_DATE_MISMATCH
ECONOMIC_SHARE_COUNT_UNRESOLVED
STALE_DENOMINATOR_TRANSPORT_FORBIDDEN
MARKET_CAP_DATE_MISMATCH
MARKET_CAP_DENOMINATOR_DATE_MISMATCH
EV_BRIDGE_DATE_MISMATCH
EV_BRIDGE_ROLLFORWARD_UNRESOLVED
PER_SHARE_DATE_MISMATCH
RESIDUAL_DATE_AUTHORITY_AMBIGUITY
```

Any state that permits two incompatible values of `VALUATION_DATE` or `SHARE_COUNT_AS_OF_DATE` fails closed with `RESIDUAL_DATE_AUTHORITY_AMBIGUITY`.

## 10. Supersession scope

This authority supersedes only the following contradictory date clauses when it is pinned in the active Contract Set:

- `OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0` Section 3:
  - `VALUATION_DATE = REFERENCE_PRICE_DATE`;
  - `SHARE_COUNT_AS_OF_DATE = REFERENCE_PRICE_DATE`;
  - the sentence requiring one same denominator date for both instantaneous market capitalization and valuation-date per-share outputs.
- downstream V3 wrapper, Integration and I3-B clauses that mechanically require valuation share-count dates or bound-effective dates to equal `REFERENCE_PRICE_DATE`.

It does not supersede:
- the economic-share-count definition or bound methodology;
- `OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0` timing mechanics;
- point-in-time cutoff discipline;
- DCF cash-flow basis rules;
- Investment Policy;
- OQS/OVS/Investment Score formulas;
- Certification;
- terminal OroTitan gates;
- Fundamentals;
- Research evidence rules.

## 11. Historical firewall and replay behavior

Existing runs retain their immutable `contract_pins`, `contract_set_sha256` and `DATA_CUTOFF`.

This authority may affect a historical run only through a controlled successor run under a successor production-admissible Contract Set.

A pure methodology replay:
- preserves the parent `DATA_CUTOFF`;
- preserves issuer, security and dossier identity;
- uses explicit lineage;
- revalidates unchanged upstream artifacts only through governed hash-verified lineage;
- never reuses a parent `VALUATION_LOCK`;
- never consumes post-cutoff evidence.

No issuer-specific exception is authorized.
