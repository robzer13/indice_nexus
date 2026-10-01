# OROTITAN CHATGPT PROTOCOL — ACCEPTANCE BATTERY V1.0

Status: EXECUTED — FOUNDATION REVIEW  
Date: 2026-10-01  
Authority: `OROTITAN_CHATGPT_OPERATING_PROTOCOL_V1_FREEZE_V1.0`  
Production mutation: NONE

---

## 1. Objective

Verify whether the approved ChatGPT operating model is sufficiently protected against the failure modes that previously caused:

- repeated analytical loops;
- stale context;
- silent state drift;
- premature stage progression;
- lost work;
- evidence contamination;
- publication leakage;
- unnecessary full reruns.

This battery is intentionally split into:

```text
A. LIVE READ-ONLY INFRASTRUCTURE CHECKS
B. EXISTING CI / REGISTRY PROTECTIONS
C. FROZEN ANALYTICAL-PROTOCOL CONTROLS
D. IMPLEMENTATION GAPS REQUIRED BEFORE VERTICAL SLICE
```

No production write test is performed.

---

## 2. Live read-only preflight

Verified on the active Supabase OroTitan project:

### RLS

RLS is enabled on:

- `orotitan_runs`
- `orotitan_run_stages`
- `orotitan_artifacts`
- `orotitan_artifact_edges`
- `orotitan_run_events`
- `research_snapshots`

### Mutation RPC permissions

The observed analytical mutation functions are:

```text
anon          = NO EXECUTE
authenticated = NO EXECUTE
service_role  = EXECUTE
```

for:

- checkpoint;
- finalization;
- pause/resume/reopen;
- artifact resolution;
- publish authorization/result;
- canonical snapshot persistence.

### Immutable / append-only guards

Live triggers are present for:

- run update guard;
- run delete prevention;
- stage update guard;
- stage delete prevention;
- artifact update guard;
- artifact delete prevention;
- event mutation prevention;
- manifest supersession.

### Concurrency

Live checkpoint/finalization/reopen functions require expected:

```text
RUN STATE_VERSION
STAGE STATE_VERSION
```

and fail closed when state changed.

### Artifact resolution

The live resolver validates:

- exact artifact ID/version;
- expected SHA-256;
- authority class.

### Publication separation

Publication remains a separate controlled operation.

`SAVE OROTITAN` therefore has no direct publication semantics.

---

## 3. Acceptance result

25 critical scenarios were reviewed.

The machine-readable matrix is:

`calibration/vnext/OROTITAN_CHATGPT_PROTOCOL_ACCEPTANCE_MATRIX_001.json`

The result is intentionally not expressed as a fake global percentage.

### Already strongly protected

The current infrastructure already provides strong protection for:

- stale-state writes;
- immutable run locks;
- checkpoint vs final separation;
- durable checkpoint manifests;
- exact artifact resolution;
- append-only operational events;
- stage reopening mechanics;
- separate publish authorization;
- invalid Integration publication;
- idempotency conflicts;
- persistence provenance.

### Frozen analytical controls

The protocol now explicitly protects:

- UNKNOWN / NOT_ASSESSABLE preservation;
- user claims as hypotheses rather than evidence;
- mandatory contradictory research;
- material-change revalidation;
- cycle normalization;
- technology substitution analysis;
- anti-anchoring;
- Red Team;
- no silent post-cutoff contamination.

These are analytical rules, not database behavior.

---

## 4. Implementation gaps found

The battery shows that the project is **not yet vertical-slice ready**.

This is expected.

The main missing layer is no longer architecture. It is implementation.

### P0 — Data Contracts V2

Required to make machine-verifiable:

- Evidence;
- Conflict;
- Open Question / Gap;
- quantitative provenance;
- Company Economic DNA;
- Technology;
- Cyclicality;
- Capital Allocation;
- analytical block outputs;
- material-change markers;
- Evidence-ID references.

This closes scenarios such as:

- post-cutoff evidence ingestion;
- invalid Evidence IDs;
- issuer/competitor contradictions;
- serial-acquirer economics.

### P0 — Process Engine V2

Required for:

- block-level states;
- dependency graph;
- block reopening;
- material-change revalidation gate;
- wrong-sector-method blocking;
- routine refresh routing;
- Red Team reopening upstream blocks;
- unresolved material gaps.

### P0 — ChatGPT ↔ Supabase bridge

Required for reliable:

```text
LOAD
STATUS
CHECKPOINT
SAVE
REFRESH
```

This layer must:

- assemble minimal context;
- read exact run/artifact versions;
- detect stale state;
- produce guarded RPC payloads;
- fail closed on connector failure;
- return a save receipt.

---

## 5. Critical conclusion

The previous failure pattern was mainly:

```text
LLM / chat reasoning
→ implicit state
→ repeated prompts
→ context reconstruction
→ accidental loops
```

The new target is:

```text
CHATGPT REASONING
→ EXACT DURABLE STATE
→ CHECKPOINT / SAVE
→ GUARDED REGISTRY
→ EXPLICIT NEXT ACTION
```

The registry foundation is already materially stronger than the previous workflow.

However, maximum fluidity will only be achieved after the three missing implementation layers exist:

1. Data Contracts V2;
2. Process Engine V2;
3. ChatGPT ↔ Supabase controlled bridge.

---

## 6. Gate before UI implementation

Do not begin final UI implementation until the following are true:

```text
DATA CONTRACTS V2 = STABLE
PROCESS ENGINE V2 = STABLE
LOAD/SAVE BRIDGE = FUNCTIONAL
ONE COMPANY VERTICAL SLICE = PASS
```

Conceptual UI work remains allowed.

User-facing OroTitan remains:

```text
FRENCH-FIRST
```

Machine semantics may remain English.

---

## 7. Exact next action

```text
DESIGN_ANALYTICAL_ENGINE_V2_DATA_CONTRACTS
```

The acceptance battery therefore confirms the previously selected sequencing:

```text
TEST FOUNDATION
→ DATA CONTRACTS
→ STATE MACHINE
→ CHATGPT BRIDGE
→ VERTICAL SLICE
→ FINAL FRENCH UI
```
