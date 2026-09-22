# OROTITAN_VNEXT_PRE_STAGE_PREFLIGHT_V0.1

**Project:** OroTitan Equity Research  
**Status:** FROZEN — GATE 9 PASS  
**Methodology change:** NO  
**Depends on:** Gate 6 State Model + Gate 7 Architecture + Gate 8 Run Controller  
**AI dependency:** NONE  
**Persistence authority:** NONE / READ-ONLY

## 0. Purpose

Gate 9 implements deterministic admission control before expensive stage or sub-stage work.

The core invariant is:

```text
STALE OR INCOHERENT INPUT
=> PRE_FLIGHT = FAIL
=> DO NOT START EXPENSIVE WORK
=> REPORT EXACT FAILURE
```

The preflight does not repair state. It does not infer missing state from chat context. It does not mutate the run, stage, artifacts, Registry, shadow database, or production database.

## 1. Covered scopes

VNext Gate 9 exposes preflight admission for:

```text
RESEARCH
DEEP_DIVE
FUNDAMENTALS
VALUATION
CERTIFICATION
INTEGRATION
```

`FUNDAMENTALS`, `VALUATION`, and `CERTIFICATION` execute inside the Deep Dive stage and therefore inherit the Deep Dive stage contract boundary.

## 2. Frozen-source alignment

### Research

The frozen Research Stage Contract requires before research work:

```text
RUN_ID
COMPANY / ISSUER identity
CANONICAL_MODE / RUN_TYPE
DATA_CUTOFF
exact RESEARCH_STAGE_CONTRACT
process-contract compatibility
baseline canonical snapshot when REFRESH
available user documents / artifacts
absence of prior blocker invalidating Research
```

### Deep Dive

The frozen Deep Dive Stage Contract requires before analytical work:

```text
RUN_ID
COMPANY / ISSUER / SECURITY identity
CANONICAL_MODE / RUN_TYPE
DATA_CUTOFF
exact DEEP_DIVE_STAGE_CONTRACT
higher-authority compatibility
baseline canonical snapshot when REFRESH / ACTIVATION
exact Research artifacts
RESEARCH stage COMPLETE
READY_FOR_DEEP_DIVE = YES
analysis-input sufficiency / lock
Evidence Ledger / Conflict Ledger references as required
no artifact / contract / cutoff mismatch
```

### Integration

The frozen Integration Stage Contract requires before projection work:

```text
RUN_ID
COMPANY / ISSUER / SECURITY identity
CANONICAL_MODE / RUN_TYPE
DATA_CUTOFF
exact INTEGRATION_STAGE_CONTRACT
exact Integration Spec
exact Screener Schema
authorized I2
authorized I3-B
higher-authority compatibility
exact Deep Dive artifacts
DEEP_DIVE stage COMPLETE
READY_FOR_INTEGRATION = YES
baseline canonical snapshot for REFRESH / ACTIVATION CHECK
identity-registry state required for projection
no artifact / cutoff / contract / method / version mismatch
```

## 3. Deterministic input contract

Implementation:

```text
runtime/vnext/pre-stage-preflight.ts
```

The request pins:

```text
scope
expected RUN_ID
expected run state_version
expected stage state_version
expected issuer / security / dossier identity
expected CANONICAL_MODE
expected RUN_TYPE
expected DATA_CUTOFF
expected process version
expected Pilotage contract version
expected contract-set SHA-256
expected stage contract name / version / SHA-256
expected contract pins
expected artifacts
expected upstream FINAL manifest when applicable
prior assurance gate state
authority compatibility
blocking execution defect state
```

The preflight then compares those expectations against freshly resolved state.

## 4. Optimistic concurrency

Both run and stage versions are checked before admission:

```text
RUN.state_version   == EXPECTED_RUN_STATE_VERSION
STAGE.state_version == EXPECTED_STAGE_STATE_VERSION
```

Any mismatch is stale state and fails admission.

Gate 9 does not update either version.

## 5. Identity integrity

Identity is never inferred from ticker text, filename, conversation, or semantic similarity.

The preflight compares exact persisted identifiers.

Research requires the frozen company / issuer identity boundary.

Deep Dive, Fundamentals, Valuation, Certification and Integration additionally require exact security identity where the execution contract requires it.

Dossier identity is also pinned when present in the bootstrap.

## 6. Point-in-time integrity

```text
stored DATA_CUTOFF == expected DATA_CUTOFF
```

For:

```text
REFRESH
ACTIVATION CHECK
run_type = REFRESH
```

an exact baseline canonical snapshot must resolve.

Gate 9 does not authorize a new cutoff and never absorbs post-cutoff evidence.

## 7. Contract-set integrity

The run-level contract set is pinned by SHA-256.

Each required stage-specific contract is independently resolved by:

```text
NAME
VERSION
CONTENT SHA-256
DURABLE LOCATOR
```

For Integration, callers must include the exact Integration Spec, Screener Schema, authorized I2, authorized I3-B and other required pins in `expectedContracts`.

A missing, mismatched, stale, or unresolvable contract fails admission.

### Resolved hash semantics

For Gate 9, a resolved contract/artifact/manifest `contentSha256` is the SHA-256 established by the resolution layer for the exact resolved content.

It must not be populated by blindly echoing the expected pin.

The resolver boundary is responsible for fetching the exact durable locator and proving the resolved content hash before handing the object to the pure Gate 9 validator.

## 8. Upstream manifest integrity

Deep Dive requires the exact authoritative Research FINAL manifest.

Integration requires the exact authoritative Deep Dive FINAL manifest.

Checks include:

```text
manifest artifact ID
version
RUN_ID
upstream stage
content SHA-256
SEALED status
AUTHORITATIVE authority
AVAILABLE state
FINAL kind
durable locator
upstream stage COMPLETE
exact handoff gate name
handoff gate state = YES
```

A CHECKPOINT manifest cannot admit a downstream stage.

A human summary cannot substitute for the upstream manifest.

## 9. Required artifact integrity

Every required artifact is resolved by exact identity:

```text
ARTIFACT_ID
VERSION
RUN_ID
ARTIFACT_TYPE
CONTENT SHA-256
EXPECTED AUTHORITY STATE
SEALED status
AVAILABLE state
DURABLE LOCATOR
```

Missing, invalidated, superseded where authority is required, unavailable, hash-mismatched, wrong-run, wrong-version or unlocatable artifacts fail admission.

The caller is responsible for supplying the frozen scope-specific artifact requirements. This preserves analytical authority in the stage contracts rather than hardcoding an invented VNext analytical checklist.

## 10. Authority and blocker state

Admission also requires:

```text
PRIOR ASSURANCE GATE = PASS
AUTHORITY STATE COMPATIBLE = YES
NO BLOCKING EXECUTION DEFECT
STAGE blocker count = 0
```

These are execution controls only. They do not decide business quality or investment merit.

## 11. Run-state firewall

Expensive work is refused when the run is already:

```text
READY_TO_PUBLISH
PUBLISHED
CANCELLED
```

This prevents an already-terminal or publication-bound run from silently re-entering analytical work.

Controlled reopening remains a separate explicit operation under frozen Registry semantics.

## 12. Scope-specific stage-state admission

```text
RESEARCH      -> RESEARCH stage
DEEP_DIVE     -> DEEP_DIVE stage
FUNDAMENTALS  -> DEEP_DIVE stage
VALUATION     -> DEEP_DIVE stage
CERTIFICATION -> DEEP_DIVE stage
INTEGRATION   -> INTEGRATION stage
```

Deep Dive sub-scopes require the stage to be `IN_PROGRESS`.

Top-level stage preflight admits only appropriate non-terminal stage lifecycle states and never treats `COMPLETE` as permission for silent rerun.

## 13. Failure reporting

Gate 9 returns a deterministic list of failure codes.

Representative classes:

```text
RUN_ID_MISMATCH
RUN_STATE_VERSION_STALE
STAGE_STATE_VERSION_STALE
IDENTITY_*_MISMATCH
DATA_CUTOFF_MISMATCH
CONTRACT_SET_HASH_MISMATCH
CONTRACT_PIN_MISSING
CONTRACT_VERSION_MISMATCH
CONTRACT_HASH_MISMATCH
CONTRACT_LOCATOR_MISSING
BASELINE_SNAPSHOT_REQUIRED
UPSTREAM_MANIFEST_*
UPSTREAM_STAGE_NOT_COMPLETE
UPSTREAM_HANDOFF_NOT_YES
ARTIFACT_*
PRIOR_ASSURANCE_GATE_FAILED
AUTHORITY_STATE_INCOMPATIBLE
BLOCKING_EXECUTION_DEFECT
```

The assertion wrapper converts any failed report into:

```text
VNEXT_PRE_STAGE_PREFLIGHT_FAIL
```

with exact failure details.

## 14. Deterministic assurance coverage

Implementation tests:

```text
tests/vnext-pre-stage-preflight.test.ts
```

The test matrix proves:

```text
clean Research admission                              PASS
clean Deep Dive admission                             PASS
clean Fundamentals admission                          PASS
clean Valuation admission                             PASS
clean Certification admission                         PASS
clean Integration admission                           PASS
stale run state_version                               REJECT
stale stage state_version                             REJECT
identity drift                                        REJECT
data-cutoff drift                                     REJECT
contract-set drift                                    REJECT
stage-contract version/hash drift                     REJECT
contract locator missing                              REJECT
checkpoint upstream manifest                          REJECT
upstream stage not COMPLETE                           REJECT
handoff != YES                                        REJECT
upstream hash drift                                   REJECT
upstream locator missing                              REJECT
artifact version/hash drift                           REJECT
artifact INVALIDATED                                  REJECT
artifact authority mismatch                           REJECT
artifact unavailable                                  REJECT
artifact locator missing                              REJECT
missing refresh baseline                              REJECT
prior assurance failure                               REJECT
authority incompatibility                             REJECT
blocking execution defect                             REJECT
missing required upstream manifest                    REJECT
unexpected upstream manifest                          REJECT
terminal/publication-bound run                        REJECT
```

## 15. Gate 9 acceptance matrix

```text
G9-01 exact RUN_ID                                  PASS
G9-02 identity pinning                              PASS
G9-03 CANONICAL_MODE / RUN_TYPE pinning            PASS
G9-04 DATA_CUTOFF pinning                          PASS
G9-05 exact run contract-set SHA                   PASS
G9-06 exact stage contract pin                     PASS
G9-07 required contract resolution                 PASS
G9-08 contract locator requirement                 PASS
G9-09 run optimistic concurrency                   PASS
G9-10 stage optimistic concurrency                 PASS
G9-11 baseline snapshot when required              PASS
G9-12 exact upstream FINAL manifest                PASS
G9-13 upstream stage COMPLETE                      PASS
G9-14 upstream handoff = YES                       PASS
G9-15 exact artifact ID/version/run/type           PASS
G9-16 artifact SHA-256 pin                         PASS
G9-17 artifact SEALED                              PASS
G9-18 artifact authority                           PASS
G9-19 artifact AVAILABLE                           PASS
G9-20 durable artifact locator                     PASS
G9-21 prior assurance gate                         PASS
G9-22 authority compatibility                      PASS
G9-23 blocker firewall                             PASS
G9-24 terminal/publication-bound run firewall      PASS
G9-25 all six required scopes covered              PASS
G9-26 deterministic failure codes                  PASS
G9-27 read-only execution                          PASS
G9-28 no ChatGPT / model dependency                PASS
G9-29 production mutation                          NONE
G9-30 shadow DB mutation                           NONE
G9-31 full VNext CI                                PASS
```

## 16. Freeze record

```text
GATE                              = 9
RESULT                            = PASS
IMPLEMENTATION                    = runtime/vnext/pre-stage-preflight.ts
TESTS                             = tests/vnext-pre-stage-preflight.test.ts
VALIDATED_IMPLEMENTATION_HEAD     = 16a9490a22723b48e0c48991927762ded2c70b0f
CI_RUN                            = 39
CI_RESULT                         = SUCCESS
PRODUCTION_MUTATION               = NONE
SHADOW_DB_MUTATION                = NONE
AI_DEPENDENCY                     = NONE
ANALYTICAL_METHODOLOGY_CHANGE     = NONE
```

V0.1 is frozen as the Gate 9 pre-stage admission authority.

Any semantic change requires a new explicit version. Silent weakening of a preflight condition is forbidden.

## 17. Out of scope

Gate 9 does not:

- certify stage outputs;
- mark a stage COMPLETE;
- repair an invalid artifact;
- retry external failures;
- classify recoveries;
- execute analytical modules;
- invoke an LLM;
- publish;
- mutate production.

Those responsibilities belong to later gates.
