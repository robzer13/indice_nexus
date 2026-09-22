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


---

## 18. Orthogonal VNext state vector

The Registry state machine above remains the persistent orchestration authority.

VNext additionally exposes eight orthogonal state dimensions so that technical execution, analytical uncertainty, price conditions, decisions and assurance cannot collapse into one overloaded status:

```text
RUNTIME_STATE
STAGE_STATE
ANALYTICAL_STATE
EVIDENCE_STATE
VALUATION_RELIABILITY
PRICE_CONDITION
DECISION_STATE
AUDIT_STATUS
```

These dimensions are not substitutes for the frozen V1 analytical contracts. They are a normalized VNext execution/read-model layer.

### 18.1 RUNTIME_STATE

```text
IDLE
READY
RUNNING
PAUSED
BLOCKED
RECOVERING
FAILED
COMPLETE
```

Legal transitions:

```text
IDLE       -> READY
READY      -> RUNNING | BLOCKED
RUNNING    -> PAUSED | BLOCKED | RECOVERING | FAILED | COMPLETE
PAUSED     -> READY | RUNNING | BLOCKED
BLOCKED    -> READY | RECOVERING | FAILED
RECOVERING -> READY | RUNNING | BLOCKED | FAILED
FAILED     -> RECOVERING
COMPLETE   -> READY
```

This state is technical only. `RUNTIME_STATE = BLOCKED` never means `DECISION_STATE = REJECT`.

### 18.2 STAGE_STATE

This is an alias of the frozen stage lifecycle vocabulary:

```text
NOT_STARTED
IN_PROGRESS
PAUSED
BLOCKED
COMPLETE
```

Its legal transitions are exactly those in section 3. `COMPLETE -> IN_PROGRESS` requires controlled reopen.

### 18.3 ANALYTICAL_STATE

This preserves the frozen execution-metadata vocabulary for analytical blocks:

```text
INSUFFICIENT
IN_PROGRESS
PROVISIONALLY_STABLE
LOCKED
```

Legal transitions:

```text
INSUFFICIENT         -> IN_PROGRESS
IN_PROGRESS          -> INSUFFICIENT | PROVISIONALLY_STABLE | LOCKED
PROVISIONALLY_STABLE -> INSUFFICIENT | IN_PROGRESS | LOCKED
LOCKED               -> IN_PROGRESS | INSUFFICIENT only through controlled reopen
```

`LOCKED != CERTIFIED`.

### 18.4 EVIDENCE_STATE

```text
UNKNOWN
SUFFICIENT
PARTIAL_BUT_DECISIONABLE
INSUFFICIENT
CONFLICTED
```

The four assessed values preserve the frozen Data Sufficiency vocabulary. `UNKNOWN` is a pre-evaluation sentinel only and is non-punitive.

Any evidence state may move to any different evidence state when new evidence, contradiction, withdrawal, basis repair or point-in-time refresh changes the evidence set.

### 18.5 VALUATION_RELIABILITY

Canonical analytical values remain:

```text
HIGH
MEDIUM
LOW
NOT_ASSESSABLE
```

VNext additionally permits:

```text
UNKNOWN
```

only as a pre-evaluation or explicitly invalidated execution sentinel. It is not a new V1 canonical valuation output.

Legal transitions:

```text
UNKNOWN -> HIGH | MEDIUM | LOW | NOT_ASSESSABLE
ASSESSED VALUE -> any different assessed value
ASSESSED VALUE -> UNKNOWN only through explicit invalidation
```

A high-quality business may legitimately have `VALUATION_RELIABILITY = NOT_ASSESSABLE`.

### 18.6 PRICE_CONDITION

```text
UNKNOWN
NOT_ASSESSABLE
ABOVE_REQUIRED_RETURN_PRICE
AT_OR_BELOW_REQUIRED_RETURN_PRICE
AT_OR_BELOW_STRONG_RETURN_PRICE
AT_OR_BELOW_EXCEPTIONAL_RETURN_PRICE
```

This is a factual market-price-to-Price-Ladder state, not an investment recommendation.

Any value may move to any different value when market price or a validated valuation ladder changes.

If `VALUATION_RELIABILITY = NOT_ASSESSABLE`, a priced return-zone condition is invalid.

### 18.7 DECISION_STATE

VNext normalizes the frozen `NEXT_ACTION` vocabulary as:

```text
UNKNOWN
INVESTABLE_NOW
WAIT_FOR_PRICE
WAIT_FOR_EVIDENCE
REFRESH_REQUIRED
REJECT
```

`UNKNOWN` is a pre-decision sentinel only.

All non-REJECT decisions may be revised when validated evidence, valuation, price or refresh state changes.

`REJECT` retains causal memory. It may transition only to `REFRESH_REQUIRED`, and only when the frozen rejection-reopen doctrine is satisfied:

```text
REVERSIBILITY = YES
+
NEW MATERIAL EVIDENCE
DIRECTLY ADDRESSES
REJECTION_REASON
```

Direct `REJECT -> INVESTABLE_NOW` is illegal.

### 18.8 AUDIT_STATUS

```text
NOT_RUN
IN_PROGRESS
PASS
FAIL
STALE
```

Legal transitions:

```text
NOT_RUN     -> IN_PROGRESS
IN_PROGRESS -> PASS | FAIL
PASS        -> STALE
FAIL        -> IN_PROGRESS
STALE       -> IN_PROGRESS
```

Audit status is deterministic assurance metadata. It is not Certification and not business quality.

## 19. Cross-state invariants

The following invariants are deterministic in VNext v0.1:

1. Technical blockage does not imply analytical rejection.
2. Evidence `UNKNOWN`, `INSUFFICIENT` or `CONFLICTED` is not a business-quality penalty.
3. `VALUATION_RELIABILITY = NOT_ASSESSABLE` does not imply weak business quality.
4. `INVESTABLE_NOW` requires decisionable evidence, assessable valuation and price support at or below an applicable return threshold.
5. Price alone cannot repair a causal `REJECT`.
6. `AUDIT_STATUS = PASS` does not mean Certification.
7. `ANALYTICAL_STATE = LOCKED` does not mean Certification.
8. Any guarded reopen is explicit and testable.
9. No state dimension is inferred from chat memory.
10. No VNext state authorizes production publication while shadow mode is active.

## 20. Machine-readable implementation

```text
runtime/vnext/state-machine.ts
  -> frozen Registry/run/stage transition model

runtime/vnext/state-model.ts
  -> orthogonal eight-domain VNext state vector

schemas/vnext/state-model-v0.1.schema.json
  -> machine-readable state-vector vocabulary

tests/vnext-state-machine.test.ts
  -> Registry/run/stage deterministic transition tests

tests/vnext-state-model.test.ts
  -> orthogonal state-domain transition and invariant tests
```

## 21. Gate 6 extended acceptance matrix

In addition to G6-01 through G6-29:

```text
G6-30 eight orthogonal state domains exist
G6-31 runtime blockage is independent from investment decision
G6-32 frozen stage lifecycle is preserved exactly
G6-33 frozen analytical execution states are preserved exactly
G6-34 UNKNOWN evidence is non-punitive
G6-35 evidence sufficiency may improve or deteriorate
G6-36 valuation reliability preserves HIGH / MEDIUM / LOW / NOT_ASSESSABLE
G6-37 valuation UNKNOWN is execution-only and requires explicit invalidation after assessment
G6-38 price condition is separate from decision state
G6-39 REJECT causal reopen requires explicit authorization
G6-40 audit PASS must become STALE before rerun
G6-41 INVESTABLE_NOW cross-state prerequisites are deterministically validated
G6-42 state-vector schema validates the eight dimensions
```

Gate 6 is complete only when both the Registry state-machine tests and orthogonal state-model tests pass in VNext CI.
