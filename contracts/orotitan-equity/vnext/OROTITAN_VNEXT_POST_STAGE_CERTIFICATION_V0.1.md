# OROTITAN_VNEXT_POST_STAGE_CERTIFICATION_V0.1

**Project:** OroTitan Equity Research  
**Status:** FROZEN — GATE 10 PASS  
**Methodology change:** NO  
**Depends on:** Gates 6, 7, 8, 9  
**AI dependency:** NONE  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

Gate 10 makes stage completion a certified deterministic operation.

A stage may not become `COMPLETE` merely because an executor claims that work is finished.

The required protocol is:

```text
READ CURRENT STATE
-> CERTIFY PREPARED FINAL BUNDLE
-> IF FAIL: DO NOT MUTATE
-> IF PASS: ATOMIC FINALIZATION BOUNDARY
-> FRESH REREAD
-> VERIFY PERSISTED FINAL STATE
```

This preserves the frozen artifact-first, registry-second finalization doctrine.

## 1. Frozen authority preserved

The Registry / Manifest V1 authority requires:

```text
finish required stage work
run stage self-audit
allocate artifact IDs / versions
generate final bytes
compute SHA-256
build FINAL Stage Manifest
persist immutable bytes
verify stored bytes
then and only then finalize Registry state atomically
```

Gate 10 begins after immutable artifact bytes are prepared and verified.

It does not implement artifact storage itself.

## 2. Certification inputs

Implementation:

```text
runtime/vnext/post-stage-certification.ts
```

Certification consumes:

```text
fresh run state
fresh stage state
expected run state_version
expected stage state_version
required output policy derived from the pinned stage scope
prepared sealed outputs
resolved exact input artifacts
prepared FINAL Stage Manifest
lineage edges
required lineage edges
stage self-audit result
contract-pin verification result
forbidden-mutation assurance result
```

The caller must derive the required output policy from the exact pinned stage contract and run scope. VNext may not silently shrink that policy.

## 3. Pre-finalization run/stage checks

Before any finalization mutation:

```text
RUN_ID exact
RUN state_version exact
run not READY_TO_PUBLISH / PUBLISHED / CANCELLED
stage RUN_ID exact
stage code exact
stage state_version exact
stage lifecycle = IN_PROGRESS
stage blocker count = 0
self-audit = PASS
contract pins = VERIFIED
forbidden mutation assurance = PASS
```

Any failure blocks the mutation boundary.

## 4. Required outputs

Every required output must resolve as a prepared immutable artifact with:

```text
exact ARTIFACT_ID + VERSION
correct RUN_ID
correct STAGE
required ARTIFACT_TYPE
artifact_status = SEALED
authority_class = AUTHORITATIVE_STAGE_OUTPUT
authority_state = AUTHORITATIVE
availability_state = AVAILABLE
registered SHA-256 == SHA-256 of resolved bytes
durable locator available
schema validation PASS when schema-bound
```

Duplicate output identities fail closed.

A missing type in the frozen required-output policy fails closed.

## 5. FINAL Stage Manifest

The manifest must satisfy the frozen V1 manifest contract.

Gate 10 checks:

```text
supported manifest_schema_version
manifest_kind = FINAL
manifest_id == registered manifest artifact_id
correct stage-manifest artifact type
exact RUN_ID
exact stage
exact stage_revision
issuer/security identity matches run
CANONICAL_MODE / RUN_TYPE match run
DATA_CUTOFF matches run
baseline snapshot matches run
process version matches run
Pilotage contract version matches run
stage contract name/version/SHA match stage
contract_set_sha256 matches run
stage_status = COMPLETE
correct stage handoff gate
handoff gate state = YES
critical blockers = []
manifest artifact SEALED
manifest artifact AUTHORITATIVE
manifest artifact AVAILABLE
manifest bytes hash-verify
manifest locator available
```

Stage-specific gates remain:

```text
RESEARCH    -> READY_FOR_DEEP_DIVE
DEEP_DIVE   -> READY_FOR_INTEGRATION
INTEGRATION -> READY_TO_PUBLISH
```

## 6. Manifest/output reconciliation

The Stage Manifest must not list itself in `output_artifacts[]`.

Every non-manifest prepared output must appear exactly in the FINAL manifest with matching:

```text
ARTIFACT_ID
VERSION
ARTIFACT_TYPE
CONTENT_SHA256
AUTHORITY_CLASS
MEDIA_TYPE
SIZE_BYTES
```

Every listed input must resolve to exact sealed and available bytes with the expected content hash.

A human summary cannot substitute for an exact artifact reference.

## 7. Lineage

Gate 10 validates the explicit artifact dependency graph.

Checks:

```text
no self-edge
no duplicate edge
all endpoints resolve
all required lineage edges exist
```

Supported V1 relation vocabulary remains:

```text
CONSUMES
DERIVED_FROM
SUPERSEDES
BASELINE_OF
REVALIDATES
```

Cross-run lineage is explicitly supported by the Gate 10 contract.

This is required for Refresh, where a current-run artifact may `REVALIDATE` or relate to a prior-run baseline artifact without rewriting historical provenance.

## 8. Mutation boundary

Only a certification result of `PASS` may call:

```text
finalizeCertifiedStage(...)
```

The persistence adapter is responsible for implementing the frozen atomic registry transaction:

```text
register / verify artifacts
register lineage edges
register FINAL manifest
set active manifest pointer
set lifecycle COMPLETE
set handoff gate
append finalization event
increment applicable state_version values
COMMIT
```

Gate 10 itself is provider-neutral and does not yet bind this interface to live Supabase.

## 9. Fresh reread

After the atomic finalization boundary, Gate 10 performs a fresh read and independently verifies:

```text
run immutable fields did not drift
stage immutable fields did not drift
stage lifecycle = COMPLETE
handoff gate = YES
active manifest pointer = exact certified FINAL manifest
stage state_version = prior version + 1
every certified output is registered
the certified manifest artifact is registered
```

A successful mutation followed by inconsistent persisted state is reported as a fresh-reread failure rather than silently accepted.

## 10. Deterministic test coverage

Implementation tests:

```text
tests/vnext-post-stage-certification.test.ts
```

Covered cases include:

```text
valid bundle -> certified finalization                 PASS
missing mandatory output                               REJECT
superseded/non-authoritative output                     REJECT
hash mismatch                                           REJECT
missing locator                                         REJECT
schema failure                                          REJECT
CHECKPOINT used as final manifest                       REJECT
manifest/output mismatch                                REJECT
manifest self-reference                                 REJECT
missing lineage                                         REJECT
duplicate lineage                                       REJECT
lineage self-edge                                       REJECT
unresolved lineage endpoint                             REJECT
cross-run REVALIDATES lineage                           PASS
post-mutation stage not COMPLETE                        DETECT
wrong active manifest pointer                           DETECT
missing registered output after mutation                DETECT
forbidden immutable drift after mutation                DETECT
self-audit failure                                      REJECT
contract-pin verification failure                       REJECT
forbidden-mutation assurance failure                    REJECT
```

## 11. Gate 10 acceptance matrix

```text
G10-01 certification occurs before COMPLETE mutation       PASS
G10-02 required outputs enforced                           PASS
G10-03 output schema assurance                             PASS
G10-04 output SHA-256 verification                         PASS
G10-05 durable locator verification                        PASS
G10-06 output authority / availability                     PASS
G10-07 FINAL manifest required                             PASS
G10-08 manifest identity and run/stage pins                PASS
G10-09 manifest contract pins                              PASS
G10-10 manifest/output exact reconciliation                PASS
G10-11 manifest self-reference firewall                    PASS
G10-12 exact input resolution                              PASS
G10-13 lineage integrity                                   PASS
G10-14 cross-run Refresh lineage                           PASS
G10-15 self-audit required                                 PASS
G10-16 forbidden-mutation assurance required               PASS
G10-17 optimistic run/stage concurrency                    PASS
G10-18 fresh reread after mutation                         PASS
G10-19 COMPLETE persisted state independently verified     PASS
G10-20 handoff YES independently verified                  PASS
G10-21 active FINAL manifest pointer verified              PASS
G10-22 registered output rows verified after finalization  PASS
G10-23 deterministic tests                                 PASS
G10-24 no model / ChatGPT dependency                       PASS
G10-25 production mutation                                 NONE
G10-26 shadow DB mutation                                  NONE
G10-27 VNext CI                                            PASS
```

## 12. Freeze record

```text
GATE                              = 10
RESULT                            = PASS
IMPLEMENTATION                    = runtime/vnext/post-stage-certification.ts
TESTS                             = tests/vnext-post-stage-certification.test.ts
VALIDATED_IMPLEMENTATION_HEAD     = 3b1b64304e3246e0a224f71c1189469b06fffdc3
CI_RUN                            = 43
CI_RESULT                         = SUCCESS
ANALYTICAL_METHODOLOGY_CHANGE     = NONE
PRODUCTION_MUTATION               = NONE
SHADOW_DB_MUTATION                = NONE
```

V0.1 is frozen as the Gate 10 post-stage certification authority.

Any semantic change requires a new explicit version.

## 13. Out of scope

Gate 10 does not:

- repair a failing bundle;
- classify recovery;
- retry providers;
- make analytical judgments;
- alter scoring;
- publish;
- mutate production.

Those responsibilities remain downstream.
