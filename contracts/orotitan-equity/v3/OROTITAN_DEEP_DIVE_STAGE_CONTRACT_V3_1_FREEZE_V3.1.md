# OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3.1 - FREEZE V3.1

**Status:** FROZEN - V3.1
**Base incorporated by reference:** OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3_FREEZE_V3.0 @ SHA256 f52932630702d4249d47da370604899c07d7aad9046fbd616945cdfccf2dbd16
**Depends on:** V3.1 Process + V3.1 Pilotage + unchanged V2 Research Stage
**Registry stage code:** DEEP_DIVE
**Methodology authorities:** OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0 + OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0 + OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0
**Scoring formula change:** NO
**Certification change:** NO

All V3.0 Deep Dive rules remain unchanged except the date-alignment clauses below.

## 1. Valuation timing tuple

Before Valuation can finalize:

```text
TIME_ORIGIN = VALUATION_DATE = DATA_CUTOFF
SHARE_COUNT_AS_OF_DATE = ECONOMIC_SHARE_COUNT_DATE = VALUATION_DATE
PER_SHARE_OUTPUT_DATE = VALUATION_DATE
EV_TO_EQUITY_BRIDGE_DATE = VALUATION_DATE when FCFF
REFERENCE_PRICE_DATE <= VALUATION_DATE
```

Any residual date ambiguity blocks Valuation.

## 2. Denominator admission

EXACT:
- exact point-in-time economic share count;
- `SHARE_COUNT_AS_OF_DATE = VALUATION_DATE`;
- complete exact bridge to the target date.

BOUNDED:
- finite positive lower/upper bounds;
- `BOUND_EFFECTIVE_DATE = VALUATION_DATE`;
- `BOUND_COMPLETENESS = COMPLETE`;
- `UNBOUNDED_MOVEMENT_CLASSES = 0`;
- exact provenance and joint-state treatment.

UNKNOWN:
- `VALUATION_ADMISSION = NO` for denominator-dependent valuation outputs;
- no `VALUATION_LOCK`.

A bound or exact count established at `REFERENCE_PRICE_DATE` is not reusable at a later `VALUATION_DATE` unless every intervening movement class is exactly closed or rigorously hard-bounded under the frozen share-count method.

## 3. Market-cap and reference-price separation

Observed market capitalization, when required, is dated to `REFERENCE_PRICE_DATE` and uses a same-date economic share count. It must not reuse the valuation-date denominator when the dates differ.

Reference price may remain earlier than the valuation date when admitted by the frozen timing/staleness rules.

## 4. EV bridge

For FCFF, a material bridge dated before `VALUATION_DATE` requires explicit cutoff-compliant roll-forward/reconciliation. If unresolved, FCFF equity value and dependent outputs are `NOT_ASSESSABLE`.

FCFE and supported Owner Earnings equity DCF do not use the ordinary FCFF bridge.

## 5. Same-cutoff methodology replay revalidation

A methodology-successor run may revalidate unchanged non-valuation upstream artifacts only when exact persisted content hashes match and the revalidation contract confirms same issuer/security/dossier, same cutoff, no new evidence and no affected non-valuation methodology.

Revalidation creates explicit lineage. It does not copy parent outputs conversationally.

Parent Valuation artifacts and parent `VALUATION_LOCK` are never revalidated.

## 6. Handoff

`READY_FOR_CERTIFICATION = YES` requires a new-run `VALUATION_LOCK` produced under this Contract Set. All existing Certification and downstream order constraints remain unchanged.
