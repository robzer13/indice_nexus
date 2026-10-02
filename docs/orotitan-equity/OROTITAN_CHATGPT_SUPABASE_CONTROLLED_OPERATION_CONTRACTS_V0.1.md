# OROTITAN_CHATGPT_SUPABASE_CONTROLLED_OPERATION_CONTRACTS_V0.1

**Project:** OroTitan Equity Research  
**Program:** post-C7 Analytical Engine V2  
**Status:** DESIGN CANDIDATE — NOT FROZEN  
**Date:** 2026-10-02  
**Methodology change:** NO  
**Scoring change:** NO  
**Valuation change:** NO  
**Production mutation:** NONE  
**Supabase migration:** NONE

---

## 0. PURPOSE

These contracts define the controlled boundary between ChatGPT and the durable OroTitan state held through Supabase / Registry infrastructure.

They sit between:

```text
CHATGPT OPERATING PROTOCOL
+ FROZEN DATA CONTRACTS V2
+ FROZEN PROCESS ENGINE V2
→ CONTROLLED OPERATION CONTRACT
→ FUTURE BRIDGE IMPLEMENTATION
→ EXISTING GUARDED REGISTRY / STORAGE
```

The contracts define what may be read, what may be proposed for mutation, what must be pinned before mutation, and what must be verified after mutation.

They do not implement the bridge.

---

## 1. AUTHORITIES

This design is subordinate to:

```text
OROTITAN_CHATGPT_OPERATING_PROTOCOL_V1_FREEZE_V1.0
OROTITAN_ANALYTICAL_DATA_CONTRACTS_V2_FREEZE_V1.0
OROTITAN_PROCESS_ENGINE_V2_FREEZE_V1.0
existing Registry migrations / guarded RPCs
existing Stage Manifest and artifact-registry contracts
existing publication authorization boundary
```

The Process Engine remains pure. This layer is the first contract allowed to translate deterministic Process Engine intent into a future infrastructure operation.

---

## 2. PRIMARY CONTROL INVARIANT

```text
CHATGPT MAY REASON FREELY
≠
CHATGPT MAY MUTATE CANONICAL STATE FREELY
```

Every durable mutation requires:

1. an explicit controlled operation type;
2. exact RUN_ID and STAGE_CODE;
3. expected run and stage state versions;
4. deterministic idempotency key;
5. deterministic request fingerprint;
6. validated persistence inputs where artifacts are involved;
7. a guarded Registry operation;
8. post-write durable-state verification.

If any prerequisite is absent or stale:

```text
FAIL CLOSED
→ RE-LOAD
→ RE-EVALUATE
```

Never blind-retry a mutation with guessed state.

---

## 3. MACHINE CONTRACT

Machine-readable envelope:

```text
schemas/vnext/chatgpt-supabase/controlled-operation.schema.v0.1.json
```

The root contract admits only:

```text
LOAD
LOAD_RESULT
CHECKPOINT_STAGE
FINALIZE_STAGE
REOPEN_STAGE
NOOP
MUTATION_RECEIPT
OPERATION_FAILURE
```

Publication is deliberately absent.

---

## 4. LOAD CONTRACT

### 4.1 LOAD is read-only

```text
LOAD OROTITAN <COMPANY>
→ resolve identity
→ resolve dossier
→ resolve active run if any
→ resolve stage
→ resolve exact contract pins
→ resolve state versions
→ resolve active manifest
→ resolve blockers
→ resolve artifact index
→ build minimal context plan
```

LOAD MUST NOT:

- create a run;
- start a stage;
- reopen a stage;
- checkpoint;
- finalize;
- publish;
- mutate an artifact.

The machine result therefore carries:

```text
mutation_allowed = false
```

### 4.2 L0 control state

When a run exists, LOAD must surface the control values needed for safe continuation:

- RUN_ID;
- RUN_STATUS;
- RUN_STATE_VERSION;
- RUN_TYPE;
- CANONICAL_MODE;
- DATA_CUTOFF;
- CONTRACT_SET_SHA256;
- CURRENT_STAGE;
- STAGE_REVISION;
- STAGE_LIFECYCLE;
- STAGE_STATE_VERSION;
- HANDOFF_GATE_STATE;
- ACTIVE_MANIFEST exact ref;
- blockers;
- artifact index;
- current process-state artifact ref where present.

No write operation may infer these values from chat memory.

### 4.3 Artifact resolution

An artifact used as authority is resolved by exact:

```text
RUN_ID
+ ARTIFACT_ID
+ VERSION
+ expected SHA-256 where pinned
+ required authority class where pinned
```

This mirrors the deployed `resolve_orotitan_artifact` guard, which rejects unavailable, unsealed, non-authoritative, hash-mismatched or authority-class-mismatched artifacts.

A storage URI alone is never sufficient authority.

### 4.4 Context assembly

LOAD returns a context plan, not an instruction to bulk-load the archive.

```text
L0 = run control context
L1 = active analytical context
L2 = supporting dossier context on demand
L3 = raw source material only when needed
```

Every loaded authoritative artifact retains exact ID/version/hash provenance.

---

## 5. SAVE / CHECKPOINT CONTRACT

Process Engine V2 supplies deterministic SAVE intent:

```text
CHECKPOINT | FINALIZE | BLOCK | NOOP
```

The bridge may translate that intent only as follows.

### 5.1 CHECKPOINT

```text
Process Engine = CHECKPOINT
→ CHECKPOINT_STAGE
→ deployed checkpoint_orotitan_stage(...)
```

Allowed target lifecycle:

```text
IN_PROGRESS | PAUSED
```

### 5.2 BLOCK

```text
Process Engine = BLOCK
→ CHECKPOINT_STAGE
→ target_lifecycle = BLOCKED
→ deployed checkpoint_orotitan_stage(...)
```

A blocker therefore remains durable without falsely completing the stage.

### 5.3 FINALIZE

```text
Process Engine = FINALIZE
→ FINALIZE_STAGE
→ deployed finalize_orotitan_stage(...)
```

The bridge does not decide finalization eligibility. It consumes the frozen Process Engine decision and then submits a bundle that must still satisfy Registry manifest locks.

### 5.4 NOOP

```text
Process Engine = NOOP
→ NOOP
→ zero persistence call
```

NOOP must not be converted into a checkpoint merely to produce activity.

---

## 6. OPTIMISTIC CONCURRENCY / CAS

Every mutating operation is pinned to:

```text
expected_run_state_version
expected_stage_state_version
```

These values come from the immediately preceding authoritative LOAD.

Registry mismatch errors are classified:

```text
RUN_STATE_VERSION_MISMATCH
STAGE_STATE_VERSION_MISMATCH
→ STALE_STATE
```

Response:

```text
DO NOT RETRY BLINDLY
→ LOAD
→ reconstruct current state
→ rerun Process Engine decision
```

The old mutation intent is stale after a concurrency conflict.

---

## 7. IDEMPOTENCY

Every mutation requires:

```text
idempotency_key
request_fingerprint_sha256
```

The same idempotency key may be replayed only with the same request fingerprint under the existing Registry idempotency guard.

A receipt with:

```text
idempotent_replay = true
```

is acceptable only after the bridge re-loads durable state and verifies that it matches the intended operation.

Changing the key to bypass a failed guarded request is prohibited.

---

## 8. ARTIFACT BYTE PERSISTENCE

The Registry controls metadata authority, but metadata must not point to bytes that were never durably persisted.

For CHECKPOINT / FINALIZE bundles:

```text
construct accepted artifact bytes
→ persist bytes to approved private backend
→ verify persistence receipt / exact-byte integrity
→ persistence_receipts_verified = true
→ invoke guarded Registry transaction
```

If byte persistence or integrity verification fails:

```text
PERSISTENCE_INTEGRITY
→ Registry mutation prohibited
```

The controlled-operation contract does not define a new storage backend and does not create a new database table.

---

## 9. MANIFEST / BUNDLE BOUNDARY

CHECKPOINT and FINALIZE transport these existing Registry inputs without redefining their internal authority:

```text
manifest
manifest_registration
output_artifacts[]
edges[]
```

The bundle must already satisfy the existing Stage Manifest and artifact-registry contracts.

This V0.1 contract deliberately treats those payloads as delegated validated objects rather than duplicating their schemas.

No controlled-operation wrapper may weaken the underlying Registry validation.

---

## 10. REOPEN CONTRACT

A completed stage may be reopened only through:

```text
REOPEN_STAGE
→ deployed reopen_orotitan_stage(...)
```

Required controls:

- exact RUN_ID;
- exact STAGE_CODE;
- expected run state version;
- expected stage state version;
- target lifecycle `IN_PROGRESS | BLOCKED`;
- structured reason;
- idempotency key;
- request fingerprint.

The existing RPC remains authoritative for downstream-stage invalidation and stage revision increments.

Published or cancelled terminal runs are not reopened in place; they require the existing successor-run path.

Block-level Process Engine freshness transitions inside an active stage do not automatically imply a stage-level REOPEN RPC.

---

## 11. POST-WRITE VERIFICATION

A successful RPC return is not sufficient to declare the bridge operation complete.

After every mutation:

```text
RPC SUCCESS / IDEMPOTENT REPLAY
→ LOAD durable state again
→ verify intended lifecycle / manifest / stage revision
→ resolve registered artifacts
→ verify artifact integrity / authority
→ verify state matches intent
→ issue MUTATION_RECEIPT
```

A valid `MUTATION_RECEIPT` requires:

```text
durable_state_reloaded = true
state_matches_intent = true
artifact_integrity_verified = true
```

If verification fails, the operation is not reported as complete.

---

## 12. ERROR CONTRACT

Errors are normalized to:

```text
STALE_STATE
NOT_FOUND
INVALID_STATE
CONTRACT_VIOLATION
PERSISTENCE_INTEGRITY
AUTHORITY_VIOLATION
INFRASTRUCTURE
```

For V0.1:

```text
retry_without_reload_allowed = false
```

This is intentional. Retry policy belongs to controlled orchestration and the frozen Process Engine retry guard, not to an opaque transport loop.

---

## 13. PUBLICATION FIREWALL

Every mutation request in this contract carries:

```text
publish_authorized = false
```

Neither:

- SAVE;
- CHECKPOINT;
- FINALIZE;
- REOPEN;
- a `READY_TO_PUBLISH` run state;
- a successful Integration finalization

authorizes publication.

Publication remains behind the separate existing:

```text
GO PUBLISH <COMPANY>
```

authority and its dedicated Registry operations.

---

## 14. SECURITY BOUNDARY

The existing critical Registry RPCs are service-role-only.

Therefore the future bridge must:

- keep service-role credentials outside the model-visible payload;
- expose only narrow controlled operations to ChatGPT;
- never expose arbitrary SQL;
- never expose arbitrary Supabase mutation;
- never accept user-supplied RPC names;
- validate every operation against this contract before invocation;
- return sanitized operation receipts, not secrets.

This design does not change RLS or production configuration.

---

## 15. FIRST VERTICAL-SLICE OPERATION SET

The minimum bridge implementation required after this design is:

```text
1. LOAD issuer / dossier / active-run control state
2. resolve exact authoritative artifact refs
3. assemble L0 + required L1 context
4. CHECKPOINT_STAGE from Process Engine CHECKPOINT/BLOCK
5. FINALIZE_STAGE from Process Engine FINALIZE
6. REOPEN_STAGE when a completed stage is legitimately reopened
7. post-write LOAD + verification receipt
8. fail-closed stale-state handling
```

Run creation, publication and broad administrative mutation are not included in the first bridge slice.

---

## 16. ACCEPTANCE TARGETS

The controlled-operation implementation must later prove at minimum:

- LOAD is mutation-free;
- no run is created by LOAD;
- exact artifact ID/version is preserved;
- hash / authority mismatch fails closed;
- missing CAS versions reject mutation;
- stale run version requires reload;
- stale stage version requires reload;
- same idempotency key + same fingerprint is safely replayable;
- conflicting idempotency replay fails;
- CHECKPOINT cannot imply stage completion;
- BLOCK becomes durable without finalization;
- FINALIZE is impossible without Process Engine FINALIZE intent and valid Registry bundle;
- NOT_STARTED / incomplete state cannot be silently promoted through the bridge;
- persistence failure prevents Registry mutation;
- post-write verification is mandatory;
- publication is never authorized by this contract;
- no arbitrary SQL/RPC surface exists.

---

## 17. IMPLEMENTATION PACKAGE FOR THIS DESIGN

```text
docs/orotitan-equity/OROTITAN_CHATGPT_SUPABASE_CONTROLLED_OPERATION_CONTRACTS_V0.1.md
schemas/vnext/chatgpt-supabase/controlled-operation.schema.v0.1.json
tests/vnext-chatgpt-supabase-controlled-operation-contracts.test.ts
calibration/vnext/OROTITAN_CHATGPT_SUPABASE_CONTROLLED_OPERATION_CONTRACTS_CANDIDATE_001.json
```

No production mutation. No Supabase migration. No bridge runtime is implemented by this design mission.

---

## 18. STATUS

```text
DATA_CONTRACTS_V2
= FROZEN_V1_0

PROCESS_ENGINE_V2
= FROZEN_V1_0

CONTROLLED_OPERATION_CONTRACTS
= DESIGN_CANDIDATE_NOT_FROZEN

NEXT
= RUN_CHATGPT_SUPABASE_CONTROLLED_OPERATION_CONTRACTS_CI
```
