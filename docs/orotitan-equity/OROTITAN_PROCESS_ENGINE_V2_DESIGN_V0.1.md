# OROTITAN_PROCESS_ENGINE_V2_DESIGN_V0.1

**Project:** OroTitan Equity Research  
**Program:** post-C7 Analytical Engine V2  
**Status:** DESIGN CANDIDATE — NOT FROZEN  
**Date:** 2026-10-02  
**Methodology change:** NO  
**Scoring change:** NO  
**Valuation change:** NO  
**Production mutation:** NONE

---

## 0. PURPOSE

Process Engine V2 is the deterministic orchestration layer between:

```text
FROZEN ANALYTICAL DATA CONTRACTS V2
→ PROCESS DECISION
→ FUTURE CHATGPT ↔ SUPABASE BRIDGE
```

It decides:

- which analytical block is executable;
- whether a blocker must be resolved first;
- which dependency cone is invalidated by a material change;
- whether a block may become analytically `LOCKED`;
- how price-only / routine-fundamental / full refresh routes reopen work;
- whether SAVE intends CHECKPOINT / FINALIZE / BLOCK / NOOP;
- whether a repeated execution is justified or is a loop;
- whether required sector overlays and dependency edges are satisfied.

It does not research, analyze, value, score, persist bytes, invoke Supabase RPCs, publish, or alter frozen methodology.

---

## 1. AUTHORITIES

This design is subordinate to:

```text
OROTITAN_CHATGPT_OPERATING_PROTOCOL_V1_FREEZE_V1.0
OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0
OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0
OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2_FREEZE_V2.0
OROTITAN_ANALYTICAL_DATA_CONTRACTS_V2_FREEZE_V1.0
existing guarded Registry / checkpoint / finalize / reopen infrastructure
```

Data Contracts V2 freeze baseline:

```text
PR #336 global reconciliation
→ c956cfbadeeaab0b36c01e242a27487fb7a433ef

PR #337 freeze
→ b18f48eae1caf9838df4a16427c501658b55e14f
```

Earlier Process Engine PR #331 is design provenance only. PR #335 was intentionally closed before merge when the dual Data Contracts authority conflict was discovered.

---

## 2. CRITICAL STATE SEPARATION

The Process Engine must preserve three independent state domains.

### 2.1 Analytical execution status

Frozen by Data Contracts V2:

```text
INSUFFICIENT
IN_PROGRESS
PROVISIONALLY_STABLE
LOCKED
```

Meaning:

- `INSUFFICIENT`: current analytical work cannot progress honestly with available input;
- `IN_PROGRESS`: analysis is active;
- `PROVISIONALLY_STABLE`: may feed downstream work but remains reopenable;
- `LOCKED`: finalized analytical block for the current run.

### 2.2 Process freshness / invalidation

Owned by Process Engine V2:

```text
CURRENT
REOPENED
STALE
```

Meaning:

- `CURRENT`: current analytical state remains usable;
- `REOPENED`: block is a direct material-change origin and requires controlled re-execution;
- `STALE`: block is downstream of a changed dependency and must be reconciled/re-executed before becoming current again.

These values never enter the analytical `execution_status` field.

### 2.3 Durable stage/save state

Existing registry / protocol semantics remain separate:

```text
Registry lifecycle:
NOT_STARTED | IN_PROGRESS | PAUSED | BLOCKED | COMPLETE

SAVE classes:
CHAT_WORKING | CHECKPOINTED | FINAL_SEALED
```

Therefore:

```text
CHECKPOINTED ≠ analytical execution status
BLOCKED ≠ analytical verdict
STALE ≠ analytical verdict
COMPLETE ≠ analytical block status
```

This separation closes the semantic conflict discovered during the post-#334 Data Contracts review.

---

## 3. PROCESS STATE CONTRACT

Machine-readable process control state is defined by:

```text
schemas/vnext/process-engine/process-state.schema.v0.1.json
```

It references, rather than duplicates:

```text
blockCode
blockExecutionStatus
payloadContext
```

from frozen Data Contracts V2.

A block process row contains:

```text
block_id
presence
analytical_execution_status
freshness
upstream_block_refs[]
critical_unresolved_gap
blocking_conflict
sector_overlays[]
material_revalidation_status
completion_audit_passed
last_execution_fingerprint
retry_count
```

Cross-field invariants that JSON Schema cannot safely express are enforced by the deterministic TypeScript engine.

---

## 4. BLOCK PRESENCE

Process Engine distinguishes:

```text
NOT_STARTED
PRESENT
```

Rules:

```text
NOT_STARTED
→ analytical_execution_status = null
→ freshness = CURRENT

PRESENT
→ analytical_execution_status must be one frozen analytical status
```

A `NOT_STARTED` block cannot silently appear as `LOCKED`.

---

## 5. DEPENDENCY GRAPH

Each block carries:

```text
upstream_block_refs[]
```

The engine may additionally consume pinned:

```text
REQUIRED_BLOCK_DEPENDENCIES[]
```

resolved by the future Bridge from authoritative method/contract state.

The engine validates:

1. no duplicate block;
2. process-state shape invariants;
3. no self-dependency;
4. every upstream block exists;
5. graph is acyclic;
6. every pinned required dependency edge is present;
7. canonical block order is the deterministic tie-breaker.

Any violation:

```text
FAIL_CLOSED
```

---

## 6. PROVISIONAL DOWNSTREAM EXECUTION VS TERMINAL LOCK

Frozen analytical semantics permit:

```text
PROVISIONALLY_STABLE upstream
→ may feed provisional downstream execution
```

but terminal completion is stricter:

```text
downstream block → LOCKED
only if every required upstream dependency
= LOCKED + CURRENT
```

This prevents provisional evidence from becoming terminal truth merely because a downstream block was already computed.

---

## 7. MATERIAL CHANGE / REOPENING

For a material change:

```text
DIRECT CHANGE ORIGIN
→ freshness = REOPENED

TRANSITIVE DOWNSTREAM DEPENDENCY
→ freshness = STALE
```

The underlying prior analytical artifact remains immutable and traceable.

The engine returns the deterministic transition plan; the future Bridge persists the new process-state version through guarded storage.

Unrelated blocks remain `CURRENT`.

---

## 8. MATERIAL CHANGE REVALIDATION GATE

The ChatGPT protocol requires six checks before a material changed conclusion can be durably finalized:

```text
1. RE-OPEN MATERIAL EVIDENCE
2. VERIFY ROOT / PRIMARY SOURCES
3. SEARCH DISCONFIRMING EVIDENCE
4. TEST BEST ALTERNATIVE EXPLANATION
5. RECONCILE AFFECTED DOWNSTREAM BLOCKS
6. RECORD WHY PRIOR STATE CHANGED
```

Process Engine deterministically maps them to:

```text
NOT_REQUIRED
PENDING
PASS
FAIL
```

It does not decide whether an economic conclusion is true. It enforces whether the required revalidation procedure is complete.

`PENDING` is executable work, not a terminal blocker.

---

## 9. SECTOR OVERLAY VALIDATION

Process Engine does not select sector methodology.

Input:

```text
REQUIRED_SECTOR_OVERLAYS[]
= pinned authoritative method-plan projection
```

For each required overlay:

- target block exists;
- exactly one matching overlay exists;
- state = `APPLIED`.

`REQUIRED_MISSING`, `CONFLICTED`, an absent overlay, a duplicate overlay, or a false `NOT_APPLICABLE` cannot satisfy a required method.

---

## 10. ANALYTICAL LOCK ELIGIBILITY

A block may transition from `PROVISIONALLY_STABLE` to `LOCKED` only if:

```text
presence = PRESENT
freshness = CURRENT
analytical status = PROVISIONALLY_STABLE or already LOCKED for validation
no critical unresolved gap
no blocking conflict
no required/conflicted overlay blocker
completion self-audit passed
material revalidation = NOT_REQUIRED or PASS
required method-plan overlays valid
required dependency edges valid
every upstream dependency = LOCKED + CURRENT
```

A persisted `LOCKED + CURRENT` block is revalidated against these conditions. If it violates them, the resolver fails closed instead of trusting a corrupted persisted state.

---

## 11. NEXT BLOCK / NEXT ACTION

Outputs:

```text
FAIL_CLOSED
RESOLVE_BLOCKER
EXECUTE_BLOCK
FINALIZE_BLOCK
NO_BLOCK_ACTION
```

Resolution order:

1. validate state shape + dependency graph;
2. validate pinned dependency requirements;
3. fail closed on inconsistent persisted `LOCKED + CURRENT` state;
4. preserve an already active executable current block;
5. surface material blockers;
6. execute the first topologically eligible unstarted/in-progress/insufficient/reopened/stale/revalidation-pending block;
7. lock the first eligible `PROVISIONALLY_STABLE` block;
8. if every block is `LOCKED + CURRENT`, return no block action;
9. otherwise surface the unresolved dependency/revalidation condition.

No progress percentage or score is created.

---

## 12. REFRESH ROUTING

The engine reuses frozen V2 refresh classes:

```text
PRICE_ONLY_DELTA
ROUTINE_FUNDAMENTAL_DELTA
FULL_REFRESH_REQUIRED
```

### PRICE_ONLY_DELTA

```text
Research = MINIMAL_REVALIDATION
Fundamentals = REVALIDATE_PRIOR_LOCK
direct REOPENED = VALUATION
downstream STALE = CROSS_BLOCK_RECONCILIATION when present
fundamental blocks preserved
OQS may change = NO
Certification = required
Integration = required
```

A price-only route declaring a fundamental block changed fails closed.

### ROUTINE_FUNDAMENTAL_DELTA

Requires at least one changed fundamental block.

```text
direct changed fundamentals → REOPENED
dependency cone → STALE
VALUATION + CROSS_BLOCK_RECONCILIATION → invalidated where present
unrelated blocks → preserved CURRENT
```

### FULL_REFRESH_REQUIRED

```text
all current analytical blocks → direct reopen set
no analytical block preserved as current truth
historical artifacts remain immutable
```

---

## 13. SAVE DISPOSITION

Process Engine decides intent only:

```text
CHECKPOINT
FINALIZE
BLOCK
NOOP
```

It does not persist.

Terminal phase boundaries:

```text
RESEARCH
CERTIFICATION / RECONCILIATION
INTEGRATION
→ may return FINALIZE when all durable prerequisites + phase gate pass
```

Intermediate Deep Dive phases:

```text
FUNDAMENTALS
VALUATION
→ CHECKPOINT
```

Registry lifecycle remains authoritative for stage lifecycle.

Every SAVE disposition has:

```text
publishAuthorized = false
```

Publication still requires separate `GO PUBLISH <COMPANY>`.

---

## 14. LOOP / RETRY GUARD

Execution fingerprint:

```text
SHA256(JSON([
  RUN_ID,
  BLOCK_ID,
  INPUT_VERSION,
  EVIDENCE_SET_HASH,
  METHOD_VERSION,
  OUTPUT_SCHEMA_VERSION
]))
```

Same fingerprint with no justified new information:

```text
NO_NEW_INFORMATION
→ RETRY DENIED
```

Allowed reasons remain explicit:

- new evidence;
- resolved conflict;
- method change;
- deterministic bug fix;
- critical user input;
- explicit forensic justification.

A bounded retry budget may still deny execution.

---

## 15. PERSISTENCE / BRIDGE BOUNDARY

Process Engine remains pure.

It does not:

- query or mutate Supabase;
- invoke checkpoint/finalize/reopen RPCs;
- resolve artifact bytes;
- write process state;
- mutate snapshot pointers;
- publish;
- use chat memory as authority.

Future Bridge:

```text
LOAD exact durable state
→ validate / adapt
→ call pure Process Engine
→ receive deterministic transition intent
→ invoke narrow guarded persistence operation
→ verify returned durable state
```

No new database table is required by this design decision. Process state may be persisted as a versioned registered execution artifact unless Bridge review proves a queryability requirement that justifies a physical schema change.

---

## 16. ACCEPTANCE TARGETS

Process Engine V2 must pass at minimum:

- T07 prior canonical conclusion overturned;
- T11 wrong sector method attempted;
- T12 material gap unresolved;
- T18 Red Team reopens earlier block;
- T19 price-only delta;
- T20 routine fundamental delta;
- T21 full refresh required;
- T23 complete-stage finalization eligibility;
- stale-state prevention;
- repeated-prompt / same-fingerprint loop prevention;
- checkpoint/final sealing separation;
- exact Data Contracts block/status namespace regression.

---

## 17. IMPLEMENTATION PACKAGE

```text
lib/orotitan-equity/post-c7/process-engine-v2.ts
schemas/vnext/process-engine/process-state.schema.v0.1.json
tests/vnext-process-engine-v2-design.test.ts
docs/orotitan-equity/OROTITAN_PROCESS_ENGINE_V2_DESIGN_V0.1.md
calibration/vnext/OROTITAN_PROCESS_ENGINE_V2_CANDIDATE_001.json
```

No production mutation. No Supabase migration.

---

## 18. STATUS

```text
DOCUMENT
= OROTITAN_PROCESS_ENGINE_V2_DESIGN_V0.1

DATA_CONTRACTS_V2
= FROZEN_V1_0

PROCESS_ENGINE_V2
= DESIGN_CANDIDATE_NOT_FROZEN

NEXT
= RUN_PROCESS_ENGINE_V2_CI_AND_REVIEW
```
