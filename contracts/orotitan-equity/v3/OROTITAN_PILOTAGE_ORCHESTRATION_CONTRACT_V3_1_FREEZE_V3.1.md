# OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3.1 - FREEZE V3.1

**Status:** FROZEN - V3.1
**Base incorporated by reference:** OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3_FREEZE_V3.0 @ SHA256 724d74ec82c5908f4d5d3c8a8062e64bfa19ced9abadee634ffd353e2ad21fa9
**Depends on:** OROTITAN_EXECUTION_PROCESS_V3_1_FREEZE_V3.1
**Methodology authority:** NONE
**Historical run rebinding:** FORBIDDEN

All V3.0 routing rules remain unless superseded here.

## 1. Active timing authority

For new V3.1 runs Pilotage requires the exact pinned `OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0` in addition to the frozen share-count and DCF-timing authorities.

Pilotage verifies, but does not construct:
- `VALUATION_DATE = DATA_CUTOFF`;
- `SHARE_COUNT_AS_OF_DATE = VALUATION_DATE`;
- reference-price separation;
- bridge-date requirements.

## 2. Controlled methodology successor

A pure methodology replay preserves the parent cutoff and identity and starts with same-cutoff non-valuation revalidation.

The normal parent route remains `ACTIVE + DEEP_DIVE + prior Contract Set`.

A `BLOCKED` parent is exceptionally admissible only through the production successor RPC when the Registry atomically proves the exact repaired conflict state defined by V3.1 Process. Pilotage must not change the parent to `ACTIVE` to satisfy admission.

## 3. Admission and handoff

Valuation remains blocked until:
- canonical timing tuple deterministic;
- denominator representation governed at `VALUATION_DATE`;
- DCF basis required inputs admitted;
- EV bridge governed at `VALUATION_DATE` where applicable.

Certification still requires a valid new-run `VALUATION_LOCK`. No parent valuation artifact or lock may cross by revalidation.

## 4. Side effects

This contract alone creates no run, mutation, publication authorization or canonical pointer movement.
