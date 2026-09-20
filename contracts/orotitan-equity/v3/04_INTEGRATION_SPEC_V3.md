# 04_INTEGRATION_SPEC_V3 — ECONOMIC SHARE COUNT UNCERTAINTY OVERLAY

**Status:** FROZEN DESIGN — V3.0
**Base:** 04_INTEGRATION_SPEC_V2 @ SHA256 475aa3255c7b4160de18fac7e59f7a02e062380de7a7b7948b5e081ad12929fd
**Method authority:** OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0
**Integration creates methodology:** NO

## 1. Purpose

V3 adds a lossless projection of the governed point-in-time economic-share-count representation. It does not decide whether a bound is lawful. Deep Dive Valuation does that under the frozen methodology authority.

The composed publication contract is:

V1 analytical core
+ V2 product overlay
+ V3 denominator-methodology overlay.

## 2. Required V3 projection

Every analysis-capable V3 snapshot carries:

v3_methodology.economic_share_count

with:
- methodology_version;
- representation = EXACT | BOUNDED | UNKNOWN;
- share_count_as_of_date;
- valuation_admission = YES | NO;
- exact_count when EXACT;
- lower_bound and upper_bound when BOUNDED;
- bound_completeness and unbounded_movement_classes when BOUNDED;
- unknown_reason when UNKNOWN;
- provenance_refs[];
- propagation_status.

The as-of date must equal the snapshot REFERENCE_PRICE_DATE for this methodology version.

## 3. Representation rules

EXACT:
- exact_count is a positive integer;
- valuation_admission = YES;
- no bound or unknown fields.

BOUNDED:
- lower_bound and upper_bound are positive integers;
- lower_bound <= upper_bound;
- bound_completeness = COMPLETE;
- unbounded_movement_classes = 0;
- valuation_admission = YES;
- exact_count absent;
- no midpoint or synthetic point estimate stored.

UNKNOWN:
- one canonical unknown_reason is required;
- valuation_admission = NO;
- no exact_count or bound endpoints.

## 4. Downstream preservation

When representation is BOUNDED, every affected canonical numeric output must preserve the range produced by the analytical authority. Integration must not:
- replace a range with midpoint/endpoints;
- round endpoints into a scalar;
- infer independence;
- reconstruct a missing denominator;
- change MOS/reliability/gate states to make the payload validate.

Existing V1 range-capable fields remain the storage location for primary expected return, normalized return where applicable, price ladder prices, OVS and Investment Score.

If Investment Score is a range:
- one class band across the whole interval -> that class;
- class-boundary crossing -> investment_class = NOT_AVAILABLE.

## 5. Certification and gate mapping

A denominator-caused MOS boundary crossing maps to NOT_ASSESSABLE.
A denominator-caused gate truth-value crossing maps to NOT_ASSESSABLE.
The existing terminal OroTitan rule remains unchanged.

UNKNOWN economic share count cannot produce a completed Valuation Lock and therefore cannot produce an integration-ready Deep Dive under this method.

## 6. Validation composition

V3 validation must:
1. validate the unchanged analytical core under the active compatible V1-core validator;
2. validate V2 product metadata;
3. validate 04_SCREENER_SCHEMA_V3;
4. verify denominator representation consistency;
5. run I2 deterministic reconciliation;
6. run history-transition checks;
7. fail closed on any range reversal, date mismatch, state mismatch or hidden scalarization assertion.

## 7. Historical compatibility

Existing V1/V2 snapshots and runs are immutable. V3 metadata is never backfilled into historical snapshots merely to make them V3-shaped.

A historical run requiring this new methodology must use a controlled successor run with a V3 Contract Set. No retroactive contract rebind is authorized.

## 8. Production boundary

This design artifact alone does not authorize a database migration, run creation, pointer update or publication.
