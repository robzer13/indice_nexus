# OroTitan Registry V1.8 — Manifest Persistence Integrity Repair

**Date:** 2026-09-20  
**Classification:** PERSISTENCE / REGISTRY DATA-INTEGRITY DEFECT  
**Governance scope:** emergency V2.0.x implementation repair  
**Analytical methodology change:** NO  
**TotalEnergies analytical change:** NONE  
**Production data migration:** NONE

## Blocking defects

1. Existing exact artifact ID/version reuse was treated as a no-op by `orotitan_insert_artifact_registration()`. Successor checkpoint activation then superseded predecessor-bound reused outputs.
2. Stage Manifest registration trusted caller-supplied `content_sha256` and size without proving them against exact persisted immutable bytes.

## Canonical repair architecture

### V1.6 — checkpoint output revalidation and successor reuse

Migration:

`migrations/20260920_orotitan_registry_v1_6_checkpoint_output_revalidation.sql`

It:
- strictly reconciles all immutable registration fields before reusing an existing artifact;
- rebinds exact reused outputs to the successor manifest before predecessor supersession;
- preserves predecessor-only supersession;
- establishes reused-output successor semantics for future checkpoints;
- provides the base repair operation for already-affected checkpoints;
- preserves immutable artifact bodies, provenance, events and lineage history.

### V1.8 — Stage Manifest exact-byte persistence receipt

Migration:

`migrations/20260920_orotitan_registry_v1_8_manifest_persistence_receipt_integrity.sql`

The trusted executor must persist the Stage Manifest first, reread the exact immutable GitHub object at the pinned commit/path, and submit those reread bytes as a persistence receipt.

The Registry recomputes from those exact bytes:
- UTF-8 byte size;
- SHA-256;
- Git blob SHA-1 (`blob <size>\0<bytes>`);
- JSON semantic equality with the manifest being registered;
- exact immutable GitHub repository/path/commit/blob/storage URI provenance.

Any mismatch raises before the manifest row, output bindings, lineage, active-manifest pointer, run/stage CAS, or checkpoint event can survive.

The database does not attempt network access to GitHub. The independently verified reread receipt is the storage verification boundary compatible with the current private GitHub artifact-store architecture.

For already-affected active checkpoints, V1.8 replaces the preliminary edge-derived V1.6 repair RPC with the canonical verified-byte form of `revalidate_orotitan_checkpoint_outputs(...)`. Repair admission is derived from the exact reread active-manifest JSON, not from a possibly incomplete historical edge inventory. The operation separately proves:

- the historical Registry SHA that was actually recorded;
- the SHA-256 of the immutable reread bytes;
- immutable size / Git blob / commit-path provenance;
- that every supplied repair target appears exactly once in `output_artifacts`;
- that type/hash/size/media/authority metadata exactly match both the active manifest and Registry row;
- that a superseded target is attributable to a predecessor actually superseded by the active checkpoint.

Only the explicitly supplied affected target subset is rebound. Unaffected already-current outputs are not rewritten.

## Historical immutability

V1.6, deployed V1.7, and additive V1.8 do not edit historical Registry rows during deployment.

In particular, deployment does not repair any TotalEnergies manifest/artifact row automatically. The defective historical manifest remains evidence and must be remediated only through a separately admitted run repair.

## Security boundary

- public checkpoint/finalization RPCs retain the established internal execution model;
- the canonical V1.8 repair RPC is `SECURITY DEFINER` with fixed `search_path = pg_catalog, public`;
- the preliminary V1.6 repair signature is removed during V1.7 deployment;
- execution is revoked from `PUBLIC`, `anon`, and `authenticated`, and granted only to `service_role`;
- V1.8 validation/bundle helpers are not directly callable by `service_role`.

## Regression matrix

The repository matrix covers:
- all-new successor outputs;
- all-reused successor outputs;
- partial reuse;
- exact reuse;
- hash mismatch rejection;
- size mismatch rejection;
- omitted successor target rejection;
- cross-run rejection;
- INVALIDATED rejection;
- unavailable rejection;
- wrong active manifest;
- stale run CAS;
- stale stage CAS;
- idempotent replay;
- idempotency-key fingerprint conflict;
- caller SHA mismatch vs verified persisted bytes;
- caller size mismatch vs verified persisted bytes;
- Git blob mismatch;
- missing persistence receipt;
- history preservation;
- critical set invariant: `M1={A,B,C,D}`, `M2={B,C,D,E}` leaves B/C/D/E current and A superseded.

## Deployment boundary

Production deployment and company-run remediation are separate control boundaries.

After deployment:
1. inspect live definitions and privileges;
2. verify V1.6 + deployed V1.7 + V1.8 are active;
3. stop implementation phase;
4. freshly re-query the target run;
5. only then admit a controlled run repair.


## V1.7 preservation / V1.8 additive sequencing

Deployed V1.7 remains immutable:

`migrations/20260920_orotitan_registry_v1_7_final_output_rebinding.sql`

V1.8 is an additive successor. It preserves V1.7's generic CHECKPOINT/FINAL successor-output rebinding, stage/revision validation, manifest-kind authority semantics, supersession behavior, and lineage handling. V1.8 adds the exact-byte persistence receipt gate and receipt-aware existing-checkpoint repair only.

The stale PR identity `20260920_orotitan_registry_v1_7_manifest_persistence_receipt_integrity.sql` is not part of the final implementation path.
