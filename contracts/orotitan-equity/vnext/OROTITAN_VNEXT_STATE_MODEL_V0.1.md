# OROTITAN_VNEXT_STATE_MODEL_V0.1

**Project:** OroTitan Equity Research  
**Status:** CANDIDATE FOR GATE 6  
**Methodology change:** NO  
**Scope:** orchestration state, stage state, admission, reopening, publication boundary

## 0. Purpose

VNext formalizes execution state as an explicit deterministic state machine.

The model must preserve the frozen V1 semantics while eliminating implicit transition logic.

Core rule:

```text
STATE CHANGE
= EXPLICIT COMMAND
+ CURRENT AUTHORITATIVE STATE
+ GUARDS
+ IDEMPOTENCY
+ OPTIMISTIC CONCURRENCY
+ APPEND-ONLY EVENT
```

A user-facing summary, conversation memory, filename, timestamp, or inferred "latest" object is never state authority.

## 1. State dimensions

The model keeps separate:

```text
RUN STATUS
STAGE LIFECYCLE
STAGE REVISION
HANDOFF GATE
ACTIVE MANIFEST KIND
CONTRACT STATUS CODE
PUBLICATION AUTHORIZATION / RESULT
```

These dimensions must not be collapsed into a single ambiguous status.

## 2. Run status vocabulary

```text
CREATED
ACTIVE
PAUSED
BLOCKED
READY_TO_PUBLISH
PUBLISHED
CANCELLED
```

Legal transitions:

```text
CREATED          -> ACTIVE | PAUSED | BLOCKED | CANCELLED
ACTIVE           -> PAUSED | BLOCKED | READY_TO_PUBLISH | CANCELLED
PAUSED           -> ACTIVE | BLOCKED | CANCELLED
BLOCKED          -> ACTIVE | PAUSED | CANCELLED
READY_TO_PUBLISH -> PUBLISHED | BLOCKED | CANCELLED
PUBLISHED        -> terminal
CANCELLED        -> terminal
```

No other transition is legal.

## 3. Stage lifecycle vocabulary

```text
NOT_STARTED
IN_PROGRESS
PAUSED
BLOCKED
COMPLETE
```

Normal legal transitions:

```text
NOT_STARTED -> IN_PROGRESS
IN_PROGRESS -> PAUSED | BLOCKED | COMPLETE
PAUSED      -> IN_PROGRESS | BLOCKED
BLOCKED     -> IN_PROGRESS | PAUSED
```

`COMPLETE -> IN_PROGRESS` is illegal as a normal transition and is legal only through the controlled `STAGE_REOPEN` operation.

## 4. Stage codes and handoff gates

```text
RESEARCH    -> READY_FOR_DEEP_DIVE
DEEP_DIVE   -> READY_FOR_INTEGRATION
INTEGRATION -> READY_TO_PUBLISH
```

Handoff gate values:

```text
NOT_EVALUATED
YES
NO
```

A handoff gate is an execution admission decision, not an analytical score.

## 5. Manifest authority

```text
CHECKPOINT
FINAL
```

Rules:

1. A CHECKPOINT manifest may support recovery but never downstream admission.
2. A downstream stage requires the exact upstream FINAL manifest.
3. A COMPLETE stage must have an authoritative FINAL manifest.
4. Reopening preserves the prior FINAL manifest as immutable historical evidence.
5. Active manifest identity is exact `artifact_id + version + hash`.

## 6. Stage admission

### 6.1 Research

Research may start only when:

```text
run_status in {CREATED, ACTIVE}
target RESEARCH lifecycle = NOT_STARTED
run is not terminal
contract pins are valid
identity requirements for the run mode are satisfied
```

Starting Research moves:

```text
RESEARCH: NOT_STARTED -> IN_PROGRESS
run_status -> ACTIVE
current_stage -> RESEARCH
```

### 6.2 Deep Dive

Deep Dive may start only when:

```text
RESEARCH.lifecycle_status = COMPLETE
RESEARCH.handoff_gate_state = YES
RESEARCH.active_manifest_kind = FINAL
exact Research FINAL manifest resolves successfully
DEEP_DIVE.lifecycle_status = NOT_STARTED
run_status not in {PUBLISHED, CANCELLED}
```

### 6.3 Integration

Integration may start only when:

```text
DEEP_DIVE.lifecycle_status = COMPLETE
DEEP_DIVE.handoff_gate_state = YES
DEEP_DIVE.active_manifest_kind = FINAL
exact Deep Dive FINAL manifest resolves successfully
INTEGRATION.lifecycle_status = NOT_STARTED
run_status not in {PUBLISHED, CANCELLED}
```

## 7. Stage completion

A stage may become COMPLETE only through finalization.

Required guards:

```text
FINAL manifest persisted and hash-verified
required output artifacts persisted and hash-verified
manifest belongs to same run and same stage
stage contract pin matches run authority
critical blockers = none
handoff gate evaluated
registry finalization transaction succeeds
expected run state_version matches
expected stage state_version matches
```

If persistence or registry reconciliation fails:

```text
stage != COMPLETE
run does not advance
```

## 8. Reopening

A COMPLETE stage may reopen only through `STAGE_REOPEN`.

Required effects:

```text
stage_revision += 1
lifecycle_status -> IN_PROGRESS or BLOCKED
handoff_gate_state -> NOT_EVALUATED
active_manifest_* -> NULL
completed_at -> NULL
prior FINAL manifest artifact remains immutable historical evidence
prior output artifacts remain immutable historical evidence
downstream eligibility invalidated
STAGE_REOPENED event appended
run.current_stage -> reopened stage
run.run_status -> ACTIVE when target lifecycle = IN_PROGRESS
run.run_status -> BLOCKED when target lifecycle = BLOCKED
```

If downstream stages exist, they are invalidated deterministically.

For a Research reopen:

```text
DEEP_DIVE -> BLOCKED
INTEGRATION -> BLOCKED
```

For a Deep Dive reopen:

```text
INTEGRATION -> BLOCKED
```

Each invalidated downstream stage receives:

```text
contract_status_code -> UPSTREAM_STAGE_REOPENED
handoff_gate_state -> NOT_EVALUATED
active_manifest_* -> NULL
completed_at -> NULL
stage_revision += 1 only when the invalidated stage had been COMPLETE
```

A PUBLISHED or CANCELLED run cannot reopen.

## 9. Pause, checkpoint and resume semantics

PAUSED means execution is intentionally suspended and can resume without resolving a blocker.

BLOCKED means an explicit condition prevents legal progression.

Operational rules:

```text
CHECKPOINT target lifecycle may be IN_PROGRESS | PAUSED | BLOCKED
CHECKPOINT always sets active_manifest_kind = CHECKPOINT
CHECKPOINT maps run status to ACTIVE | PAUSED | BLOCKED respectively
PAUSE requires current stage lifecycle = IN_PROGRESS
PAUSE requires active_manifest_kind = CHECKPOINT
RESUME requires lifecycle in {PAUSED, BLOCKED}
RESUME requires active_manifest_kind = CHECKPOINT
RESUME -> stage IN_PROGRESS + run ACTIVE
```

Neither PAUSED nor BLOCKED can admit a downstream stage.

## 10. Run READY_TO_PUBLISH

The run may enter READY_TO_PUBLISH only if:

```text
INTEGRATION.lifecycle_status = COMPLETE
INTEGRATION.handoff_gate_state = YES
INTEGRATION.active_manifest_kind = FINAL
exact Integration FINAL manifest resolves successfully
snapshot candidate resolves exactly
schema validation = PASS
I2 reconciliation = PASS
I3-B admission = PASS
no publication blocker
```

READY_TO_PUBLISH does not mutate canonical production state.

## 11. Publication

Publication requires a distinct explicit authorization.

```text
GO PUBLISH
-> record authorization
-> verify exact candidate and Integration FINAL manifest
-> invoke authorized publication path
-> record result
```

Legal outcomes:

```text
success             -> run_status = PUBLISHED
recoverable failure -> run_status remains READY_TO_PUBLISH
non-recoverable failure -> run_status = BLOCKED
```

VNext shadow execution has publication authority disabled.

## 12. Terminal invariants

```text
PUBLISHED:
- published_at must be set
- no stage mutation
- no reopening
- no successor mutation of this run

CANCELLED:
- cancelled_at must be set
- no stage mutation
- no reopening
```

Historical artifacts and events remain immutable and resolvable.

## 13. Concurrency and idempotency

Every material mutation requires:

```text
EXPECTED_RUN_STATE_VERSION
EXPECTED_STAGE_STATE_VERSION when stage-scoped
IDEMPOTENCY_KEY
REQUEST_FINGERPRINT_SHA256
```

Same idempotency key + same request fingerprint:

```text
return prior result
```

Same idempotency key + different request fingerprint:

```text
IDEMPOTENCY_CONFLICT
fail closed
```

Stale state version:

```text
CONCURRENT_STATE_CHANGE
fail closed
```

## 14. Fail-closed invariants

The following are always illegal:

```text
CHECKPOINT -> downstream admission
COMPLETE stage without FINAL manifest
handoff YES with critical blocker
Deep Dive before Research FINAL + YES
Integration before Deep Dive FINAL + YES
READY_TO_PUBLISH before Integration FINAL + YES
publication without explicit publication authorization
stage mutation after PUBLISHED
stage mutation after CANCELLED
silent contract-set change inside a run
silent DATA_CUTOFF change inside a run
silent baseline_snapshot_id replacement
silent stage reopening
direct COMPLETE -> IN_PROGRESS outside STAGE_REOPEN
production mutation from VNEXT_SHADOW
```

## 15. Event correspondence

Material state changes must append the corresponding event:

```text
RUN_CREATED
STAGE_STARTED
STAGE_CHECKPOINTED
STAGE_PAUSED
STAGE_RESUMED
BLOCKER_OPENED
BLOCKER_RESOLVED
STAGE_REOPENED
ARTIFACT_SET_SEALED
STAGE_FINALIZED
READY_TO_PUBLISH_DECLARED
PUBLISH_AUTHORIZED
PUBLISH_SUCCEEDED
PUBLISH_FAILED
RUN_CANCELLED
```

The event log is audit evidence and is not a substitute for current registry state.

## 16. Compatibility rule

VNext State Model V0.1 is intentionally compatible with the current Registry V1.11 vocabulary and table structure.

No new run status, stage lifecycle value, stage code, or handoff gate value is introduced by this contract.

Any future vocabulary expansion requires a new state-model version and explicit migration.

## 17. Gate 6 acceptance matrix

```text
G6-01 all legal run transitions accepted
G6-02 all illegal run transitions rejected
G6-03 PUBLISHED terminal
G6-04 CANCELLED terminal
G6-05 all legal normal stage transitions accepted
G6-06 COMPLETE -> IN_PROGRESS rejected normally
G6-07 COMPLETE -> IN_PROGRESS accepted only as STAGE_REOPEN
G6-08 CHECKPOINT cannot admit downstream
G6-09 Research FINAL + COMPLETE + YES admits Deep Dive
G6-10 Research NO blocks Deep Dive
G6-11 Deep Dive FINAL + COMPLETE + YES admits Integration
G6-12 Deep Dive NO blocks Integration
G6-13 Integration FINAL + COMPLETE + YES is necessary for READY_TO_PUBLISH
G6-14 publication requires explicit authorization
G6-15 reopen increments revision
G6-16 reopen resets gate to NOT_EVALUATED
G6-17 reopen clears active manifest pointer but preserves historical FINAL artifact
G6-18 upstream reopen blocks and invalidates downstream stages deterministically
G6-19 reopen maps run status to ACTIVE or BLOCKED from target lifecycle
G6-20 stale state version fails closed
G6-21 idempotency conflict fails closed
G6-22 stage and run terminal guards enforced
G6-23 VNext production-project ref rejected
G6-24 VNext shadow-project ref accepted
G6-25 model vocabulary equals frozen Registry V1.11 vocabulary
G6-26 pause requires CHECKPOINT
G6-27 resume requires CHECKPOINT
G6-28 recoverable publish failure remains READY_TO_PUBLISH
G6-29 non-recoverable publish failure becomes BLOCKED
```

Gate 6 passes only when the machine-readable model and deterministic tests implement this contract without production mutation.
