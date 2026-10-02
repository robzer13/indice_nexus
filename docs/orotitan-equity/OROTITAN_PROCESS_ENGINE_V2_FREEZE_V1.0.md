# OROTITAN_PROCESS_ENGINE_V2_FREEZE_V1.0

**Project:** OroTitan Equity Research  
**Status:** FROZEN — V1.0  
**Freeze date:** 2026-10-02  
**Reviewed baseline:** `vnext@180d817bcd522c9d1d528ce5e2521e48aadc9cce`  
**Review PR:** #338  
**Implementation head:** `c5af5f982fec2e31273e8f98c14c9d5c1c945e07`  
**Methodology change:** NO  
**Scoring change:** NO  
**Valuation change:** NO  
**Production mutation:** NONE  
**Supabase migration:** NONE

## 1. Frozen authority

The authoritative Process Engine V2 package is:

```text
lib/orotitan-equity/post-c7/process-engine-v2.ts
schemas/vnext/process-engine/process-state.schema.v0.1.json
docs/orotitan-equity/OROTITAN_PROCESS_ENGINE_V2_DESIGN_V0.1.md
```

Validation evidence is held by:

```text
tests/vnext-process-engine-v2-design.test.ts
calibration/vnext/OROTITAN_PROCESS_ENGINE_V2_CANDIDATE_001.json
```

Process Engine V2 is subordinate to the frozen Analytical Data Contracts V2 and may not rewrite their canonical analytical vocabularies.

## 2. Frozen state separation

Three state domains remain orthogonal.

### 2.1 Analytical execution status

```text
INSUFFICIENT
IN_PROGRESS
PROVISIONALLY_STABLE
LOCKED
```

### 2.2 Process freshness / invalidation

```text
CURRENT
REOPENED
STALE
```

### 2.3 Registry / save semantics

```text
Registry lifecycle:
NOT_STARTED | IN_PROGRESS | PAUSED | BLOCKED | COMPLETE

SAVE classes:
CHAT_WORKING | CHECKPOINTED | FINAL_SEALED
```

Process-control state must never be written into analytical `execution_status`.

## 3. Frozen deterministic behavior

V1.0 freezes the reviewed behavior for:

- dependency-graph validation and canonical topological ordering;
- required dependency-edge validation;
- dependency-cone reopening;
- preservation of `NOT_STARTED` rows during reopen / refresh routing;
- material-change revalidation gating;
- required sector-overlay validation;
- analytical block lock eligibility;
- blocker-aware next-block / next-action resolution;
- `PRICE_ONLY_DELTA`, `ROUTINE_FUNDAMENTAL_DELTA` and `FULL_REFRESH_REQUIRED` routing;
- SAVE disposition intent: `CHECKPOINT | FINALIZE | BLOCK | NOOP`;
- execution fingerprinting;
- bounded retry / loop guard.

Previously `LOCKED` analytical conclusions may be operationally `REOPENED` or `STALE` while retaining historical analytical provenance. Pending material revalidation is executable process work, not a new analytical status.

## 4. Validation state at freeze

```text
PR #338
= MERGED

VNext CI #804
= PASS

Screener CI #662
= PASS

BLOCKING REVIEW FINDINGS
= ADDRESSED / RESOLVED

PROCESS ENGINE V2
= FROZEN_V1_0
```

The reviewed implementation includes regression coverage for the corrected resumable-state, refresh-routing, required-overlay, payload-context and retry-budget cases.

## 5. Deliberately not owned by this freeze

Process Engine V2 does not own:

- LOAD resolution;
- ChatGPT context assembly;
- Supabase RPC invocation;
- artifact-byte persistence;
- registry transaction execution;
- snapshot publication;
- research, valuation or scoring methodology.

No new database table is required by this freeze.

## 6. Change control

Any semantic change to the frozen Process Engine V2 behavior requires an explicit successor version.

Downstream bridge code may execute Process Engine decisions but may not silently:

- merge process freshness into analytical execution status;
- bypass dependency / overlay validation;
- bypass material revalidation requirements;
- bypass retry-budget decisions;
- convert SAVE intent into publication authority;
- use chat memory as durable authority.

## 7. Next exact action

```text
DESIGN_CHATGPT_SUPABASE_CONTROLLED_OPERATION_CONTRACTS
```

Vertical slice remains `READY = NO` until the controlled ChatGPT ↔ Supabase bridge is designed, implemented and validated.
