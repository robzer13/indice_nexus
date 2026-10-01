# OROTITAN_PROCESS_ENGINE_V2_DESIGN_V0.1

**Project:** OroTitan Equity Research  
**Program:** post-C7 Analytical Engine V2  
**Status:** DESIGN CANDIDATE — NOT FROZEN  
**Date:** 2026-10-01  
**Methodology change:** NO  
**Scoring change:** NO  
**Valuation change:** NO  
**Production mutation:** NONE

---

## 0. PURPOSE

Process Engine V2 is the deterministic orchestration layer between:

```text
ANALYTICAL DATA CONTRACTS V2
→ PROCESS DECISION
→ FUTURE CHATGPT ↔ SUPABASE BRIDGE
```

It decides:

- which analytical block is executable;
- whether a blocker must be resolved first;
- which dependency cone must reopen;
- what a refresh class is allowed to reopen;
- whether a block can move from checkpointed work to terminal completion;
- whether SAVE is eligible to checkpoint, finalize, block, or do nothing;
- whether an execution retry would create a loop;
- which sector overlays required by an authoritative method plan are present and applied.

It does not perform research, analysis, valuation, scoring, persistence, RPC calls, publication, or arbitrary database mutation.

Core boundary:

```text
PROCESS ENGINE
= DETERMINISTIC STATE / TRANSITION DECISION

PROCESS ENGINE
≠ ANALYTICAL JUDGMENT
≠ STORAGE ADAPTER
≠ SUPABASE WRITER
≠ CHATGPT CONTEXT ASSEMBLER
≠ PUBLISHER
```

---

## 1. AUTHORITIES

The design remains subordinate to:

```text
OROTITAN_CHATGPT_OPERATING_PROTOCOL_V1_FREEZE_V1.0
OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0
OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0
OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2_FREEZE_V2.0
OROTITAN_ANALYTICAL_DATA_CONTRACTS_V2 reviewed package
existing guarded registry / stage / checkpoint / finalize / reopen infrastructure
```

Data Contracts V2 design review closed at:

```text
PR #329 merge
= 489c0e0f4bf41bfe75f25ee00549b927c1e99faa

review closure / continuity merge
= 8467fdabc001baf354ce3b5919a55476ee29079e
```

The stale unmerged branch `post-c7-process-engine-v2-design-001` is historical draft material only. This V0.1 candidate is based on current `vnext`.

---

## 2. GAP OWNERSHIP

Process Engine V2 owns the deterministic portion of:

```text
T07 PRIOR_CANONICAL_CONCLUSION_OVERTURNED
T11 WRONG_SECTOR_METHOD_ATTEMPTED
T12 MATERIAL_GAP_UNRESOLVED
T18 RED_TEAM_REOPENS_EARLIER_BLOCK
T19 PRICE_ONLY_DELTA
T20 ROUTINE_FUNDAMENTAL_DELTA
T21 FULL_REFRESH_REQUIRED
T23 COMPLETE_STAGE_FINALIZES — ELIGIBILITY DECISION ONLY
```

It intentionally does not own:

```text
T02 LOAD resolver / context assembler
T14 connector-failure LOAD fallback
actual SAVE persistence
guarded RPC invocation
artifact byte persistence
registry transaction execution
publication
```

Those remain ChatGPT ↔ Supabase Bridge scope.

---

## 3. BLOCK VOCABULARY

Process Engine imports the exact block vocabulary represented by Data Contracts V2:

```text
BUSINESS_MODEL
ECONOMIC_QUALITY
INDUSTRY_STRUCTURE
TECHNOLOGY
CYCLICALITY
MOAT
RUNWAY
RETURN_QUALITY
FCF_FORENSIC
CAPITAL_ALLOCATION
MANAGEMENT_GOVERNANCE
OUTSIDE_VIEW
RISK_RESILIENCE
RED_TEAM
VALUATION
CROSS_BLOCK_RECONCILIATION
```

A regression test compares this list to the JSON Schema enum.

No new analytical block is created here.

---

## 4. BLOCK STATE SEMANTICS

Supported working-state statuses remain:

```text
READY
IN_PROGRESS
CHECKPOINTED
COMPLETE
BLOCKED
NOT_ASSESSABLE
REOPENED
STALE
```

Important distinctions:

```text
CHECKPOINTED
= durable provisional work
= MAY feed downstream analysis when dependencies permit
≠ terminal completion

COMPLETE
= terminally resolved positive/normal completion

NOT_ASSESSABLE
= terminally resolved uncertainty outcome
= MAY feed downstream analysis with explicit limitation

BLOCKED
= process action required
≠ executable analytical block

STALE / REOPENED
= prior state may not remain authoritative
→ re-execution required
```

For terminal completion of a dependent block, every upstream dependency must be terminally resolved:

```text
COMPLETE
or
NOT_ASSESSABLE
```

A merely CHECKPOINTED upstream block may support provisional downstream work but not terminal downstream completion.

---

## 5. DEPENDENCY GRAPH

The per-run dependency graph is represented by each block's:

```text
upstream_block_refs[]
```

Because an omitted dependency could otherwise preserve stale downstream work, the Process Engine may also receive:

```text
REQUIRED_BLOCK_DEPENDENCIES[]
= authoritative method-plan projection
```

The engine validates that every required edge is present before using the graph for execution or refresh routing.

Process Engine validates:

1. no duplicate block;
2. no self-dependency;
3. every upstream block exists;
4. graph is acyclic;
5. deterministic topological traversal uses the canonical block order as tie-breaker.

If graph validation fails:

```text
FAIL_CLOSED
→ NO TRANSITION PLAN
```

### 5.1 Reopen cone

A material change produces an explicit two-part plan:

```text
DIRECTLY CHANGED BLOCKS
→ REOPENED

TRANSITIVE DOWNSTREAM DEPENDENCIES
→ STALE
```

The union is the affected dependency cone. Unrelated blocks are preserved.

This is the default algorithm for:

- Red Team reopening;
- routine fundamental refresh;
- material conclusion change.

The process engine does not reopen the entire dossier merely because one block changed.

---

## 6. MATERIAL CHANGE REVALIDATION

Data Contracts V2 carries:

```text
NOT_REQUIRED
PENDING
PASS
FAIL
```

Process rule:

```text
PENDING
→ terminal block completion forbidden

FAIL
→ terminal block completion forbidden

PASS
→ transition may continue if all other conditions pass
```

The analytical executor remains responsible for deciding that a change is material.

The Process Engine only enforces the recorded revalidation state.

This preserves the frozen six-part revalidation gate without asking the state machine to make economic judgments.

---

## 7. SECTOR OVERLAY VALIDATION

Process Engine does not invent sector methodology.

Input:

```text
REQUIRED_SECTOR_OVERLAYS[]
= authoritative method-plan projection
```

For every required overlay:

1. target block must exist;
2. exactly one matching overlay must exist;
3. its state must be `APPLIED`.

Therefore:

```text
REQUIRED_MISSING
CONFLICTED
NOT_APPLICABLE
absent
duplicate
```

cannot satisfy a requirement that the authoritative method plan marks as required.

This closes the machine-validation portion of wrong-sector-method prevention while keeping the economic selection authority outside this deterministic engine.

---

## 8. BLOCK COMPLETION GATE

A block is not eligible for terminal completion if any of the following holds:

```text
BLOCKED status
STALE status
critical unresolved gap
open / blocking conflict
required or conflicted sector overlay
completion self-audit not passed
material revalidation PENDING
material revalidation FAIL
required method-plan overlay absent / not APPLIED
upstream dependency not terminally resolved
```

The completion self-audit is the process projection of the frozen analytical completion standard.

Data Contracts V2 separately validates evidence IDs, evidence relevance, conflict/gap semantics and causal-link integrity.

---

## 9. NEXT BLOCK / NEXT ACTION

The resolver returns one of:

```text
FAIL_CLOSED
RESOLVE_BLOCKER
EXECUTE_BLOCK
FINALIZE_BLOCK
NO_BLOCK_ACTION
```

Rules:

1. empty or invalid dependency state → `FAIL_CLOSED`;
2. authoritative required dependency edges must be present;
3. if the current block is still executable, preserve it before surfacing an unrelated later blocker;
4. a blocker on the current block is surfaced immediately;
5. otherwise surface the first topological blocker before ordinary new work;
6. otherwise choose the first topologically executable block;
7. CHECKPOINTED blocks are finalized only when terminal prerequisites pass;
8. all COMPLETE / NOT_ASSESSABLE → no further block action.

There is no artificial percentage progress.

---

## 10. REFRESH ROUTING

The engine reuses the frozen V2 refresh classes:

```text
PRICE_ONLY_DELTA
ROUTINE_FUNDAMENTAL_DELTA
FULL_REFRESH_REQUIRED
```

and the existing V2 route semantics.

### 10.1 PRICE_ONLY_DELTA

```text
RESEARCH = MINIMAL_REVALIDATION
FUNDAMENTALS = REVALIDATE_PRIOR_LOCK
REOPEN = VALUATION + CROSS_BLOCK_RECONCILIATION when present
PRESERVE = fundamental blocks
OQS MAY CHANGE = NO
CERTIFICATION = REQUIRED
INTEGRATION = REQUIRED
```

A PRICE_ONLY_DELTA that declares a fundamental block changed fails closed.

### 10.2 ROUTINE_FUNDAMENTAL_DELTA

Requires at least one changed fundamental block.

```text
RESEARCH = TARGETED_DELTA
FUNDAMENTALS = REOPEN_AFFECTED_BLOCKS
DIRECT REOPEN = changed fundamental block(s)
STALE = downstream dependency cone
      + VALUATION
      + CROSS_BLOCK_RECONCILIATION
PRESERVE = unrelated prior work
OQS MAY CHANGE = YES, only after authorized reanalysis
```

The route cannot originate from VALUATION or CROSS_BLOCK_RECONCILIATION alone.

### 10.3 FULL_REFRESH_REQUIRED

```text
RESEARCH = FULL
FUNDAMENTALS = FULL
REOPEN = all analytical blocks
PRESERVE = no analytical block as current truth
VALUATION / CERTIFICATION / INTEGRATION = REQUIRED
```

Historical artifacts remain immutable; "reopen all" means analytical current-state invalidation, not destructive overwrite.

---

## 11. SAVE ELIGIBILITY

Process Engine decides only the intended disposition:

```text
CHECKPOINT
FINALIZE
BLOCK
NOOP
```

It does not execute persistence.

Pre-finalization decision prerequisites:

```text
stage self-audit passed
schema validation passed
identity / version checks passed
analytical reconciliation passed
required artifacts present
artifact persistence verified
no critical blocker
phase gate = YES
```

`registry reconciliation` is deliberately **not** a precondition here. The guarded checkpoint/finalize RPC performs the authoritative registry-bundle transaction; the future Bridge must verify the returned reconciled state after the RPC. Requiring post-RPC reconciliation before deciding to invoke that RPC would create a circular transition.

If durability is incomplete:

```text
CHECKPOINT
```

not false finalization.

Lifecycle guards:

```text
NOT_STARTED → NOOP
PAUSED      → CHECKPOINT / RESUME REQUIRED
BLOCKED     → BLOCK
COMPLETE    → NOOP
IN_PROGRESS → evaluate normal SAVE disposition
```

If `phaseGate = NO` without a true blocker:

```text
CHECKPOINT
```

not `BLOCKED`.

Terminal stage boundaries eligible for `FINALIZE`:

```text
RESEARCH
CERTIFICATION  → Deep Dive finalization
INTEGRATION    → READY_TO_PUBLISH finalization only
```

Intermediate:

```text
FUNDAMENTALS
VALUATION
→ CHECKPOINT
```

Every result carries:

```text
publishAuthorized = false
```

Publication remains separate.

---

## 12. EXECUTION FINGERPRINT / LOOP GUARD

Fingerprint:

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

If the fingerprint changed:

```text
EXECUTION MAY PROCEED
```

If unchanged:

retry requires an explicit allowed reason from the target architecture:

- new evidence;
- resolved conflict;
- changed method;
- corrected deterministic bug;
- user-supplied critical input;
- explicit forensic justification.

A separately supplied bounded retry budget may still forbid the retry.

No arbitrary numeric retry threshold is created in this design.

Same fingerprint + no justification:

```text
NO_NEW_INFORMATION
→ DO NOT REPEAT
```

---

## 13. STORAGE / BRIDGE BOUNDARY

This engine is deliberately pure.

It may consume normalized values corresponding to durable state, but it does not:

- query Supabase;
- write Supabase;
- call checkpoint/finalize/reopen RPCs;
- resolve artifact bytes;
- mutate current snapshot pointers;
- publish;
- update Vercel;
- use chat memory as authority.

The future Bridge will:

```text
LOAD DURABLE STATE
→ ADAPT TO PROCESS ENGINE INPUT
→ CALL PURE PROCESS ENGINE
→ VALIDATE DECISION
→ INVOKE NARROW GUARDED RPC
→ VERIFY RESULT
```

---

## 14. IMPLEMENTATION PACKAGE

Candidate package:

```text
lib/orotitan-equity/post-c7/process-engine-v2.ts
tests/vnext-process-engine-v2-design.test.ts
docs/orotitan-equity/OROTITAN_PROCESS_ENGINE_V2_DESIGN_V0.1.md
calibration/vnext/OROTITAN_PROCESS_ENGINE_V2_CANDIDATE_001.json
```

No migration is required.

---

## 15. ACCEPTANCE CRITERIA

Before the Process Engine V2 candidate can be considered stable:

1. block vocabulary matches Data Contracts V2 exactly;
2. dependency cycles fail closed;
3. authoritative required dependency edges must be present before graph-based routing;
4. dependency closure distinguishes directly REOPENED blocks from STALE downstream blocks and does not touch unrelated blocks;
5. blocked blocks are surfaced as blockers, not selected for ordinary execution;
6. current executable block is not interrupted by an unrelated later blocker;
7. NOT_ASSESSABLE is accepted as a terminally resolved upstream state;
8. CHECKPOINTED may feed provisional downstream work but cannot satisfy terminal dependency completion;
9. material revalidation PENDING / FAIL blocks completion;
10. required sector overlay missing/wrong state blocks completion;
11. PRICE_ONLY_DELTA cannot mutate fundamental blocks;
12. PRICE_ONLY_DELTA preserves OQS-affecting fundamentals;
13. ROUTINE_FUNDAMENTAL_DELTA requires a fundamental origin;
14. routine refresh reopens only affected + downstream blocks plus valuation/reconciliation;
15. FULL_REFRESH_REQUIRED reopens the full analytical set;
16. SAVE from NOT_STARTED / COMPLETE is NOOP and PAUSED does not terminally finalize;
17. SAVE with incomplete pre-finalization durability checks checkpoints;
18. registry reconciliation is verified after guarded RPC, not required circularly before it;
19. SAVE phase gate NO checkpoints unless a real blocker exists;
20. only Research / Certification / Integration can return FINALIZE;
21. no SAVE disposition authorizes publication;
22. identical execution fingerprint cannot loop without an allowed retry reason;
23. retry budget can fail closed;
24. existing VNext and Screener CI remain green;
25. no production mutation occurs.

---

## 16. DESIGN STATUS

```text
DOCUMENT = OROTITAN_PROCESS_ENGINE_V2_DESIGN_V0.1
STATUS = DESIGN_CANDIDATE_REVIEW_PATCHED
FROZEN = NO
PRODUCTION_MUTATION = NO

STALE_DRAFT_BRANCH
= post-c7-process-engine-v2-design-001
= SUPERSEDED / UNMERGED

ACTIVE_BRANCH
= post-c7-process-engine-v2-design-002

NEXT
= RUN_PROCESS_ENGINE_V2_CI_AND_REVIEW
```
