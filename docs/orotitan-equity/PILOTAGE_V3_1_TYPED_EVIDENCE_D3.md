# OroTitan Pilotage V3.1 — D3 typed evidence gates

Status: REVIEW-ONLY, part of draft PR #383 in robzer13/indice_nexus. No deployment, contract change, database mutation, or publication.

## Verified scope

Module: lib/orotitan-equity/v3/pilotage-typed-gates-v3-1.ts

This validator is invoked only by the D2 trusted loader after immutable artifact bytes, storage locators, registration SHA-256, manifest schema, declared output references and parent manifest have been checked.

The validator follows the frozen V2 Deep Dive runtime, V2 Deep Dive Stage Contract, V3.1 Deep Dive Stage Contract, and V3.1 Valuation Date Alignment Authority.

### Fundamentals checkpoint

- Require exactly one byte-verified JSON FUNDAMENTALS_LOCK in the active checkpoint/final bundle.
- Require exact run ID, data cutoff, stage revision, lock self-reference (UUID, version, SHA-256), and current FINAL Research manifest reference.
- Independently resolve all required taxonomy, business description, scoring-input and evidence-lineage references to uniquely verified artifact registrations.
- Require exact frozen V2 lock structure; Red Team status COMPLETE, a separately byte-verified FUNDAMENTAL_RED_TEAM_RECORD, and ready_for_valuation=YES.
- Expose only READY_FOR_VALUATION and FUNDAMENTAL_RED_TEAM_COMPLETE as *formal lock-gate attestations*. The qualitative merits of Red Team conclusions still require professional analytical certification.

### Valuation checkpoint / FINAL

- Require exact new-run VALUATION_LOCK and current Fundamentals Lock hash/version, data cutoff and stage revision.
- Resolve the valuation artifact, calculation ledger, assumption register, full-precision inputs and evidence lineage through exact verified references.
- Require ready_for_certification=YES **in the lock**, but do not elevate this to a routing decision without missing methodology and economic-provenance controls.
- When an embedded valuation_date_alignment input exists in the verified JSON VALUATION_ARTIFACT, validate exact field types, date consistency and the frozen V3.1 deterministic alignment function. This proves the *internal date consistency* of the declared inputs, not the external truth of denominator movements, bounds, DCF basis or bridge economics.
- A FINAL Deep Dive bundle without an exact Valuation Lock is refused.

### I2 and I3-B

The repository contains the frozen analytical computation and I3-B writer contracts and the required report artifact names, but **does not establish a standalone machine-verifiable PASS attestation for persisted report bodies**. Report presence or a self-reported PASS cannot supply independent recalculation. D3 therefore refuses to derive I2_PASS, I3B_PASS, CANONICAL_CANDIDATE_ADMITTED, or publication authorization. The source deterministic calculation and composed V3.1 snapshot validation must be wired in a later distinct implementation after verifying their input projection and reconciliation contracts.

## Hard boundaries

- D3 never creates or publishes an investment assessment; no numerical interpretation or guesswork.
- D3 never calls Supabase or GitHub directly; reads are through D2 verified content.
- D3 never writes, publishes, reconstructs missing proof from summaries, rebinds historical runs or implicitly accepts a stale checkpoint.
- Publication gate and all D1/D2 mutating functions remain inaccessible.
- Some D1 transitions remain BLOCKED until verified denominator economics, FCFF bridge/DCF basis, certified reconciliation and I3-B computation are implemented. This is intentional.
- Source files under contracts/ remain byte-for-byte unchanged.

## Tests

tests/equity-v3-pilotage-typed-gates.test.ts covers valid locks, exact reference identities, invalid prior manifest, corrupted evidence hash, missing Red Team record, lifecycle-ready states, date disagreement, unknown denominator, and explicit I2/I3-B refusal.

CI: Screener CI (lint, typecheck, tests, PostgreSQL regression and build). A successful CI run is a code-quality signal, not live-run acceptance.

Next non-production task: implement independently recomputed denominator/DCF inputs and I2/I3-B validations, then run an authenticated *read-only* smoke test against a selected V3.1 run before considering any transition endpoint.
