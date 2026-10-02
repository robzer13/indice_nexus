# OROTITAN_CHATGPT_SUPABASE_CONTROLLED_OPERATION_CONTRACTS_FREEZE_V1.0

**Project:** OroTitan Equity Research  
**Status:** FROZEN — V1.0  
**Freeze date:** 2026-10-02  
**Reviewed baseline:** `vnext@27d28d298de31c686a15aed74085d9531df565cb`  
**Review PR:** #340  
**Reviewed head:** `95b47ceeeaceed92dc7004553f3207415d55cdfc`  
**Methodology change:** NO  
**Scoring change:** NO  
**Valuation change:** NO  
**Production mutation:** NONE  
**Supabase migration:** NONE

## 1. Frozen authority

The authoritative controlled-operation contract package is:

```text
docs/orotitan-equity/OROTITAN_CHATGPT_SUPABASE_CONTROLLED_OPERATION_CONTRACTS_V0.1.md
schemas/vnext/chatgpt-supabase/controlled-operation.schema.v0.1.json
```

Validation evidence is held by:

```text
tests/vnext-chatgpt-supabase-controlled-operation-contracts.test.ts
calibration/vnext/OROTITAN_CHATGPT_SUPABASE_CONTROLLED_OPERATION_CONTRACTS_CANDIDATE_001.json
```

This package is subordinate to the frozen ChatGPT Operating Protocol, Analytical Data Contracts V2 and Process Engine V2.

## 2. Frozen operation surface

V1.0 admits only:

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

Publication operations are deliberately absent.

## 3. Frozen control invariants

### 3.1 LOAD is mutation-free

```text
LOAD
→ resolve identity / dossier / run / stage / contract pins / blockers / artifact refs
→ return mutation_allowed = false
```

LOAD does not create, start, reopen, checkpoint, finalize or publish a run.

### 3.2 Current-stage consistency

When an active run exists:

```text
LOAD_RESULT.current_stage
=
LOAD_RESULT.stage.stage_code
```

When no run exists, run control fields and stage state remain null.

A contradictory control snapshot fails closed.

### 3.3 Optimistic concurrency

Every mutation is pinned to:

```text
expected_run_state_version
expected_stage_state_version
```

A version mismatch invalidates the mutation intent and requires a fresh LOAD plus re-evaluation.

### 3.4 SAVE disposition to Registry lifecycle

```text
CHECKPOINT
→ CHECKPOINT_STAGE
→ target_lifecycle = IN_PROGRESS | PAUSED

BLOCK
→ CHECKPOINT_STAGE
→ target_lifecycle = BLOCKED

FINALIZE
→ FINALIZE_STAGE

NOOP
→ zero persistence call
```

These mappings may not be silently crossed.

### 3.5 Reopen

A completed stage may be reopened only through the guarded stage reopen operation with exact run/stage versions, structured reason, idempotency key and request fingerprint.

Block-level Process Engine freshness does not automatically imply a stage-level reopen RPC.

### 3.6 Idempotency

Every mutation requires:

```text
idempotency_key
request_fingerprint_sha256
```

Replay receipts are internally consistent:

```text
SUCCESS
→ idempotent_replay = false

IDEMPOTENT_REPLAY
→ idempotent_replay = true
```

### 3.7 Artifact authority and persistence

Authoritative artifacts are resolved by exact run/artifact/version identity with hash and authority-class checks where pinned.

For artifact-producing mutations:

```text
persist exact bytes
→ verify persistence receipt / integrity
→ invoke guarded Registry transaction
```

If persistence integrity is not verified, Registry mutation is prohibited.

### 3.8 Post-write verification

A successful mutation is not complete until durable state is reloaded and verified.

A valid receipt requires:

```text
durable_state_reloaded = true
state_matches_intent = true
artifact_integrity_verified = true
```

### 3.9 Publication firewall

Every controlled mutation carries:

```text
publish_authorized = false
```

SAVE, CHECKPOINT, FINALIZE, REOPEN or READY_TO_PUBLISH state does not authorize publication.

## 4. Validation state at freeze

```text
PR #340
= MERGED

VNext CI #809
= PASS

Screener CI #664
= PASS

REVIEW FINDING — current_stage / stage.stage_code coupling
= FIXED / RESOLVED

REVIEW FINDING — BLOCK/CHECKPOINT lifecycle coupling
= FIXED / RESOLVED

REVIEW FINDING — receipt replay consistency
= FIXED / RESOLVED

CONTROLLED OPERATION CONTRACTS
= FROZEN_V1_0
```

## 5. Security boundary

The future bridge must:

- keep service-role credentials outside model-visible payloads;
- expose only the frozen controlled operation surface;
- never expose arbitrary SQL;
- never expose arbitrary Supabase mutation;
- never accept user-selected RPC names;
- validate requests before infrastructure invocation;
- return sanitized receipts rather than secrets.

This freeze changes no RLS policy, database schema or production configuration.

## 6. Deliberately not implemented here

This freeze does not implement:

- the ChatGPT ↔ Supabase bridge runtime;
- service-role credential handling;
- run creation;
- publication;
- arbitrary administrative mutation;
- a new database table or migration.

## 7. Change control

Any semantic change to this frozen operation surface or its safety invariants requires an explicit successor version.

The bridge may implement these contracts but may not silently weaken:

- current-stage coupling;
- CAS/version checks;
- disposition/lifecycle coupling;
- idempotency;
- persistence verification;
- post-write verification;
- artifact authority checks;
- publication isolation.

## 8. Next exact action

```text
IMPLEMENT_CHATGPT_SUPABASE_BRIDGE
```

Vertical slice remains `READY = NO` until the narrow bridge is implemented and validated end-to-end.
