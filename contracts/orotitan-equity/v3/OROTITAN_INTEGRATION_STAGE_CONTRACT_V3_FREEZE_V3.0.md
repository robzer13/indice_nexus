# OROTITAN_INTEGRATION_STAGE_CONTRACT_V3 — FREEZE V3.0

**Status:** FROZEN — V3.0
**Base incorporated by reference:** OROTITAN_INTEGRATION_STAGE_CONTRACT_V2_FREEZE_V2.0 @ SHA256 74e1a3954ac3db42ccb9daa9fdc16a156eb6d7fc42f26ec522ba550451b841d0
**Depends on:** V3 Process + V3 Pilotage + V3 Deep Dive
**Projection authority:** 04_INTEGRATION_SPEC_V3 + 04_SCREENER_SCHEMA_V3
**Deterministic authority:** I2_CANONICAL_COMPUTATION_V1.1
**Admission authority:** I3B_VALIDATED_SNAPSHOT_WRITER_V1.1
**Methodology authority:** NONE

All V2 Integration controls remain except statements that the V2 projection and unchanged I2/I3-B are the terminal authorities.

## 1. Admission

Integration still requires:
- DEEP_DIVE_STAGE_STATUS = COMPLETE;
- READY_FOR_INTEGRATION = YES;
- active Deep Dive manifest = FINAL;
- exact final artifact set available and hash verified;
- exact Contract Set match.

A Valuation CHECKPOINT never admits Integration.

## 2. V3 projection

Integration projects the exact certified denominator representation into the V3 methodology overlay.

It may map and validate:
- EXACT denominator state;
- BOUNDED denominator state with exact lower/upper bounds, effective date, completeness and provenance;
- UNKNOWN only where the source analytical state is legitimately non-analysis-capable.

Integration may not create, tighten, widen, midpoint, round or otherwise reinterpret a denominator bound.

## 3. I2 V1.1

Persisted full-precision expected-return ranges and other scoring inputs must reconcile exactly to I2 V1.1.

Investment Class handling for score ranges follows I2 V1.1. Integration may not select one endpoint.

## 4. I3-B V1.1

I3-B V1.1 validates the composed V1 analytical core + V2 product overlay + V3 methodology overlay and the denominator boundary invariants.

Any date mismatch, reversed range, hidden scalarization, state inconsistency or deterministic mismatch blocks admission.

## 5. Completion and publication

All existing V2 completion, CAS, idempotency, immutable history and explicit GO PUBLISH requirements remain unchanged.

This contract authorizes no production migration or publication by itself.
