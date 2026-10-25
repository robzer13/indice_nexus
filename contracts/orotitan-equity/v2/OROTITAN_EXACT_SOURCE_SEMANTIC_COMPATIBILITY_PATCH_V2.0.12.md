# OroTitan V2.0.12 — Exact Certified Literal Compatibility

Status: additive code review candidate; **NOT yet activated as production admission authority**.
Parent: V2.0.11, lossless thesis presentation compatibility.

## Blocking, generalizable defect

The Deep Dive stage certifies source judgments using two exact source literals that the V1-compatible canonical schema does not admit:

- `analytical_metrics.return_interpretability = MEDIUM`
- `l4_operational_state.refresh_class = ACTIVATION_CHECK_OR_TARGETED_DELTA_IF_PRICE_ZONE_REACHED`

The source is FICO's frozen FINAL Deep Dive run, cutoff 2026-09-19. Integration may not silently substitute `INTERPRETABLE_WITH_LIMITATIONS` or `NOT_APPLICABLE` without altering source meaning.

## Scope and invariants

New opt-in schema builder appends **exactly those two literals** to their two field-local enumerations after checking the exact frozen V2.0.10 enum lists. It leaves all frozen V1/V2 schema bytes and the 13-pin set unchanged. No financial or terminal-gate rule changes. V2.0.11 separately preserves full certified thesis strings.

Before application, `sealed-literal-source-gate.ts` requires verifiable Registry-authorized immutable Fundamentals Lock and Readiness sources, exact run/security/issuer/dossier/cutoff identity, matching UTF-8 SHA-256 and byte lengths, and literal equality with the proposed candidate. Caller MUST check artifact IDs, registered hashes and FINAL lineage against the live Registry, not trust candidate-supplied hashes.

FICO's Fundamentals Lock is a SEALED/AVAILABLE `CHECKPOINT_STAGE_OUTPUT` with `SUPERSEDED` authority state; FINAL Deep Dive explicitly references it with valid lock lineage. Readiness is a SEALED/AVAILABLE `AUTHORITATIVE_STAGE_OUTPUT`. A superseded source is admitted only through explicit FINAL historical lineage, never by claiming it is the latest stage authority.

## Proofs, admission and publication

Synthetic end-to-end ANALYZE tests exercise the actual composed application validator and sealed-source gate. Negative tests reject unknown enum values, changed identity, changed cutoff, hash/size mismatches, or canonical rewrites; the legacy validator continues to reject the new literals.

This is a code compatibility candidate, not a company-specific analytical exception. For a run to use it, independent operator admission of source semantics, recorded exact implementation/schema SHA, full immutable candidate validation, official I2, history, I3-B, eight persisted artifacts and verified Registry finalization are still mandatory. A separate `GO PUBLISH FICO` is required before snapshot promotion.

**Never infer `READY_TO_PUBLISH` merely from CI success or a Git merge.**
