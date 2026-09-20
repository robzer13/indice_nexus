# OROTITAN_RUNTIME_PATCH_V2.0.2

**Status:** ACTIVE RUNTIME AUTHORITY UPDATE  
**Applies to:** OroTitan V2 runs created after activation of the DCF Timing Contract Set  
**Global methodology authority added:** `OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0`  
**Historical run mutation:** FORBIDDEN  
**Publication authorization:** UNCHANGED

## 1. Purpose

Runtime V2.0.2 activates the newly frozen global DCF timing authority for newly created runs while preserving every pre-existing run's immutable Contract Set.

The active Contract Pin Pack becomes V2.0.1 and contains fourteen logical pins, including:

```text
dcf_timing
= OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0
```

The active Contract Set fingerprint is:

```text
d933717b9da01e8565a3e8116ff77582ffa7ded109649ec370a0f7839eecc71a
```

## 2. Historical firewall

A run created under:

```text
1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e
```

remains pinned to that historical Contract Set. Runtime V2.0.2 must never rewrite its `contract_pins` or `contract_set_sha256`.

## 3. Controlled same-cutoff methodology successor

A successor is a lineage relationship, not a legal `RUN_TYPE`.

For an unpublished parent with no canonical snapshot:

```text
current_snapshot_id = NULL
=> legal RUN_TYPE = INITIAL
=> baseline_snapshot_id = NULL
=> parent_run_id = exact predecessor RUN_ID
```

The successor preserves the parent `DATA_CUTOFF` exactly and uses the newly active Contract Set.

This route is admitted only for an explicit methodology-migration reason, currently:

```text
DCF_TIMING_METHODOLOGY_REPLAY
```

## 4. Stage routing

The controlled successor begins with:

```text
FIRST_REGISTRY_STAGE = RESEARCH
```

because Registry stage order and exact artifact authority remain mandatory.

Research in this route is revalidation/persistence work only. No post-cutoff evidence or new company conclusion is admitted.

After exact upstream non-valuation artifacts are revalidated into the successor lineage:

```text
FIRST_ANALYTICAL_PHASE = VALUATION
```

The parent Valuation Artifact and parent Valuation Lock are not successor analytical inputs and may not be revalidated across the timing-methodology boundary.

## 5. CAS and identity firewall

Before successor creation, Pilotage must freshly re-query the parent and require exact equality on:

```text
RUN_ID
state_version
run_status
current_stage
issuer_id
security_id
dossier_id
DATA_CUTOFF
baseline_snapshot_id
contract_set_sha256
published_at
cancelled_at
```

Any mismatch is fail-closed.

The production RPC remains `create_orotitan_run`; `parent_run_id` is populated only for the controlled successor route. No `SUCCESSOR` run type is introduced.

## 6. Cross-run artifact authority

Cross-run reuse requires explicit hash-verified lineage using Registry edge semantics such as `REVALIDATES` or `DERIVED_FROM`.

Permitted in the DCF timing replay route:

```text
Research evidence/ledger artifacts unchanged by the timing method
Fundamentals analytical outputs unchanged by the timing method
FUNDAMENTALS_LOCK unchanged by the timing method
routing/provenance manifests needed to prove the above
```

Forbidden:

```text
parent VALUATION_ARTIFACT as successor analytical authority
parent VALUATION_LOCK as successor analytical authority
parent valuation-derived scoring/certification output
post-cutoff evidence
silent semantic coercion
```

## 7. Production boundary

This patch authorizes no automatic successor creation and no publication.

```text
CREATE SUCCESSOR
= requires an explicit execution bootstrap after authority/regression/hash verification

GO PUBLISH
= remains separately required
```
