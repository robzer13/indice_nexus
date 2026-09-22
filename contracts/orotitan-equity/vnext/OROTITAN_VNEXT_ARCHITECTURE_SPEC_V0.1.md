# OROTITAN_VNEXT_ARCHITECTURE_SPEC_V0.1

**Project:** OroTitan Equity Research  
**Status:** FROZEN — GATE 7 PASS  
**Methodology change:** NO  
**Depends on:** `OROTITAN_VNEXT_STATE_MODEL_V0.1`  
**Gate 6 reference commit:** `233578cd329c997ed98981426782966593d7aa69`

## 0. Purpose

This specification defines the VNext execution architecture.

It governs orchestration, module boundaries, authority, recovery, state transfer, shadow isolation, and publication controls.

It does not redefine analytical methodology, scoring, valuation conventions, Certification, OroTitan terminal logic, or sector-specific analytical semantics.

Core rule:

```text
VNext architecture
= execution machinery around frozen analytical authority
!= new analytical authority
```

## 1. Architectural layers

The VNext stack is separated into nine logical layers:

```text
L0 USER INTENT
L1 PILOTAGE / ROUTING
L2 RUN CONTROLLER
L3 STAGE CONTROLLER
L4 MODULE EXECUTION
L5 DETERMINISTIC VALIDATION / COMPUTATION
L6 ARTIFACT PERSISTENCE
L7 OPERATIONAL REGISTRY
L8 INTEGRATION / SNAPSHOT CANDIDATE
L9 PUBLICATION FIREWALL
```

### L0 User intent

User intent may authorize:

- run execution through the ordinary GO path;
- explicit publication through `GO PUBLISH <COMPANY>`.

User prose is not analytical or registry authority.

### L1 Pilotage / routing

Pilotage:

- resolves intent;
- resolves exact issuer/run state;
- resolves exact contract versions;
- resolves exact persisted artifacts;
- prepares stage bootstraps;
- reports blockers and next action.

Pilotage must not create analytical conclusions, repair analytical defects, or infer state from conversation memory.

### L2 Run Controller

The Run Controller owns deterministic run-state orchestration:

- create/resume/reopen/cancel operations;
- optimistic concurrency;
- idempotency;
- current-stage pointer;
- run-status transition enforcement.

Gate 8 will implement this controller. Gate 7 defines its boundary only.

### L3 Stage Controller

The Stage Controller owns:

- stage admission;
- checkpointing;
- pause/resume;
- finalization;
- controlled reopening;
- handoff-gate enforcement;
- stage revision and active-manifest pointer state.

The authoritative persisted stage state remains the Registry.

### L4 Module Execution

Modules perform bounded work inside a stage.

A module may be:

```text
RESEARCH
ANALYTICAL
DETERMINISTIC
CERTIFICATION
INTEGRATION
ASSURANCE
```

A module never becomes a new top-level authority layer.

### L5 Deterministic Validation / Computation

Deterministic code owns only logic explicitly designated deterministic by frozen authority.

Examples include:

- schema validation;
- exact formula execution;
- invariant checking;
- state-machine validation;
- artifact hash checking;
- deterministic gates where frozen contracts already authorize them.

Deterministic code must not convert analyst judgment into an invented formula.

### L6 Artifact Persistence

Artifact persistence is immutable-byte authority.

Required sequence:

```text
PREPARE
-> PERSIST BYTES
-> VERIFY HASH
-> REGISTER
```

Registry state may never claim successful persistence before bytes and hashes resolve.

### L7 Operational Registry

Supabase Registry is the operational source of truth for:

- run identity;
- current run status;
- current stage state;
- exact artifact identity/version;
- active manifest pointers;
- event audit trail;
- contract pins;
- concurrency versions.

The Registry stores routing state and provenance, not duplicate analytical narrative bodies.

### L8 Integration / Snapshot Candidate

Integration is a technical canonicalization layer.

It:

- resolves exact authoritative Deep Dive outputs;
- builds the canonical candidate;
- runs schema validation;
- runs pinned deterministic reconciliation;
- runs I3-B admission;
- emits pre-publication state.

Integration may not silently repair upstream analytical defects.

### L9 Publication Firewall

Publication is a distinct authority boundary.

```text
READY_TO_PUBLISH
!= PUBLISHED
```

Only explicit publication authorization may invoke the authorized publication path.

In VNext shadow mode:

```text
PUBLICATION AUTHORITY = DISABLED
```

## 2. Authority hierarchy

All execution components obey:

```text
1. Frozen analytical methodology
2. Locked investment policy / execution patches / I2 / I3-B
3. Frozen execution process
4. Frozen Pilotage / Stage Contracts
5. VNext State Model
6. VNext Architecture Spec
7. Run-specific pinned contract set and bootstrap
8. Module implementation
9. Conversational instruction
```

A lower layer may specialize execution but must fail closed on conflict with a higher layer.

## 3. Stage architecture

The company-analysis execution stages remain:

```text
RESEARCH
DEEP_DIVE
INTEGRATION
```

Their admission and transition rules are exactly those in the VNext State Model.

No VNext module may skip an upstream stage gate by direct invocation.

## 4. Module Contract

Every executable VNext module must declare one Module Contract before live use.

Minimum fields:

```text
module_contract_schema_version
module_id
module_version
module_kind
stage_code
purpose

authority:
  reads[]
  writes[]
  judgment_authority
  deterministic_authority
  forbidden_actions[]

inputs:
  required_artifact_types[]
  required_exact_references
  required_stage_state
  required_contract_pins
  data_cutoff_policy

dependencies:
  module_ids[]
  dependency_mode

execution:
  status_vocabulary
  can_parallelize
  checkpoint_policy
  retry_policy
  fail_closed

outputs:
  artifact_types[]
  authority_class
  persistence_required
  hash_required

uncertainty:
  unknown_policy
  not_applicable_policy
  missing_policy
  assumption_policy

recovery:
  recoverable_failures[]
  reopen_triggers[]
  recovery_action

environment:
  allowed_environment
  allowed_supabase_project_ref
  production_write_allowed
```

A Module Contract is routing/execution authority only. It cannot redefine a canonical analytical verdict vocabulary.

## 5. Module kinds and authority

### 5.1 RESEARCH

May:

- collect sources;
- normalize evidence;
- maintain Research ledgers/registers;
- assess execution-layer input sufficiency;
- declare Research handoff readiness under the frozen Research contract.

Must not:

- emit final Deep Dive score/valuation/OroTitan verdict.

### 5.2 ANALYTICAL

May:

- perform a bounded frozen Deep Dive analytical block;
- use analyst judgment explicitly allowed by the methodology;
- add normalized evidence and calculations;
- expose dependencies and reopen triggers.

Must not:

- override deterministic formulas;
- silently reinterpret missing/unknown evidence;
- bypass Certification.

### 5.3 DETERMINISTIC

May:

- execute pinned formulas, schemas, invariant checks and deterministic reconciliations.

Must not:

- create analyst judgment;
- choose among economically ambiguous interpretations unless the frozen contract already specifies the selector.

### 5.4 CERTIFICATION

May:

- audit completeness, consistency, traceability and score permission under frozen Certification authority.

Must not:

- improve a weak analytical outcome by rewriting evidence or assumptions.

### 5.5 INTEGRATION

May:

- map certified outputs into canonical snapshot shape;
- run schema/I2/I3-B checks.

Must not:

- repair analytical defects by silently modifying upstream conclusions.

### 5.6 ASSURANCE

May:

- test architecture, state machine, contracts, determinism, migrations and environment boundaries.

Must not:

- write production analytical state.

## 6. Deep Dive module execution status

For Deep Dive analytical execution only, preserve the existing execution-management vocabulary:

```text
INSUFFICIENT
IN_PROGRESS
PROVISIONALLY_STABLE
LOCKED
```

These are execution states, not Certification states and not analytical verdicts.

VNext legal execution transitions:

```text
INSUFFICIENT          -> IN_PROGRESS
IN_PROGRESS           -> INSUFFICIENT | PROVISIONALLY_STABLE | LOCKED
PROVISIONALLY_STABLE  -> INSUFFICIENT | IN_PROGRESS | LOCKED
LOCKED                -> IN_PROGRESS only through controlled module reopen
```

A `PROVISIONALLY_STABLE` module may feed allowed downstream work only when its declared material dependencies are satisfied.

`LOCKED != CERTIFIED`.

## 7. Deep Dive dependency graph

Canonical dependency order remains:

```text
0  IDENTITY / POINT-IN-TIME LOCK
1  EVIDENCE LEDGER
2  BUSINESS / SEGMENT / ECONOMIC MODEL
3  ECONOMIC QUALITY SYNTHESIS
4  MOAT PROOF
5  GROWTH / RUNWAY
6  RETURN ON CAPITAL / MARGINAL RETURN
7  FCF / OWNER EARNINGS / FORENSIC
8  CAPITAL ALLOCATION
9  MANAGEMENT / GOVERNANCE
10 OUTSIDE VIEW / BASE RATES
11 RISK / RESILIENCE
12 RED TEAM / PRE-MORTEM
13 VALUATION / EXPECTED RETURN
14 RESEARCH CERTIFICATION
15 SCORING
16 OROTITAN TERMINAL GATE
17 READINESS / NEXT ACTION
```

This is a dependency order, not a blanket serial scheduler.

Parallelization is allowed where declared dependencies permit it.

Hard constraints:

```text
FINAL VALUATION
requires material fundamental inputs sufficiently stable

CERTIFICATION
requires required analytical work + valuation state

SCORING
requires Certification score permission

TERMINAL OROTITAN GATE
requires its frozen certification/scoring/valuation inputs
```

No dependency engine may replace analyst judgment for which blocks a routine fundamental refresh materially reopens.

## 8. Standard analytical module envelope

An analytical module must be able to emit at minimum:

```text
BLOCK_ID
BLOCK_VERSION
EXECUTION_STATUS
EXECUTION_CONFIDENCE

CANONICAL_VERDICT_FIELDS[]
CORE_FINDINGS[]
SUPPORTING_EVIDENCE_IDS[]
CONTRADICTING_EVIDENCE_IDS[]
CALCULATION_IDS[]
MATERIAL_ASSUMPTION_IDS[]
CONFLICT_IDS[]
UNRESOLVED_POINTS[]

MATERIAL_DEPENDENCIES[]
REOPEN_TRIGGERS[]

RATIONALE
LAST_RESEARCH_DATE
DATA_CUTOFF
```

`CANONICAL_VERDICT_FIELDS[]` may contain only frozen analytical vocabulary.

Execution confidence is metadata only:

```text
HIGH | MEDIUM | LOW
```

It is never a score input unless a future higher-authority contract explicitly authorizes that change.

## 9. UNKNOWN / MISSING / N/A discipline

Architecture must preserve the following distinctions:

```text
UNKNOWN
!= MISSING
!= NOT_APPLICABLE
!= NOT_ASSESSABLE
!= INSUFFICIENT_DATA
```

Rules:

1. UNKNOWN remains UNKNOWN until evidence changes it.
2. MISSING identifies absent required material.
3. NOT_APPLICABLE requires a valid semantic reason.
4. NOT_ASSESSABLE means the method cannot responsibly produce the requested assessment.
5. INSUFFICIENT_DATA is an evidence sufficiency outcome, not a negative business verdict.
6. A module may not manufacture an assumption solely to reach completion.
7. Every material assumption must be explicit and traceable.
8. Serialization must preserve these states rather than coercing them to null, zero, false, or an invented estimate.

## 10. Research insufficiency and user assistance

For required Research inputs:

```text
SUFFICIENT
= mandatory coverage
+ evidence adequacy
+ no material blocking gap
```

When reasonably available sources are exhausted and a critical required input remains unavailable:

```text
Research -> BLOCKED_INSUFFICIENT_INPUT
READY_FOR_DEEP_DIVE -> NO
```

When a critical inaccessible input requires user assistance:

```text
Research stage -> PAUSED
durable CHECKPOINT required for recoverability
user assistance requested
same run resumes after accepted input is persisted
```

Other modules must not route around that blocked Research gate.

## 11. Recovery architecture

Recovery is state-based, not conversation-based.

Every recoverable operation must use:

- exact `RUN_ID`;
- exact stage;
- exact `state_version`;
- exact active CHECKPOINT or FINAL manifest identity;
- exact contract pins;
- deterministic idempotency key;
- request fingerprint.

Recovery families:

```text
RETRY SAME OPERATION
RESUME PAUSED/BLOCKED STAGE
REOPEN COMPLETE STAGE
CREATE SUCCESSOR RUN
```

Selection rules:

- transient failure with unchanged request: retry idempotently;
- paused/blocked stage with valid CHECKPOINT: resume;
- analytical contradiction affecting a completed stage: controlled reopen;
- material contract/cutoff change or terminal historical run: successor run.

Never mutate historical artifacts to simulate recovery.

## 12. Failure propagation

A module failure must be classified before any state mutation.

Minimum execution classes:

```text
RECOVERABLE
BLOCKING
CONTRACT_MISMATCH
PERSISTENCE_FAILURE
ENVIRONMENT_VIOLATION
TERMINAL
```

These are infrastructure diagnostics, not analytical verdicts.

Rules:

- fail closed on contract mismatch;
- fail closed on environment violation;
- persistence failure cannot produce COMPLETE;
- upstream reopening invalidates downstream eligibility;
- a downstream analytical defect routes to the appropriate upstream stage rather than being repaired in Integration.

## 13. Artifact and manifest authority

Downstream work resolves exact persisted inputs using:

```text
RUN_ID
ARTIFACT_ID
VERSION
STATUS
SHA256
MANIFEST MEMBERSHIP
```

Forbidden authority shortcuts:

```text
conversation memory
human summary
latest filename
folder scan
semantic similarity
newest timestamp without registry authority
```

CHECKPOINT is recoverability authority only.

FINAL is required for downstream stage admission.

## 14. Shadow mode

VNext development environment:

```text
OROTITAN_ENVIRONMENT = VNEXT_SHADOW
SUPABASE PROJECT REF = awgsurdyvsyolcgpnygh
```

Forbidden production project:

```text
cugpgtzygqqlxetyeven
```

Shadow invariants:

1. no production Registry mutation;
2. no production canonical snapshot mutation;
3. no production `current_snapshot_id` mutation;
4. no production secret required by VNext runtime;
5. all VNext persistence targets shadow infrastructure;
6. test fixtures must be synthetic or explicitly copied into shadow under an authorized later test-data procedure;
7. absence of production data is not an error in shadow mode.

## 15. Publication ban in VNext

Until a later explicit release gate:

```text
OROTITAN_PUBLICATION_ENABLED = false
```

The VNext runtime must treat all publication operations as forbidden even if a run reaches `READY_TO_PUBLISH`.

Allowed:

```text
build candidate
validate candidate
simulate publication preflight
record shadow assurance evidence
```

Forbidden:

```text
promote production snapshot
mutate production current_snapshot pointer
call production canonical writer
represent shadow candidate as production current state
```

## 16. Refresh architecture

Refresh classification remains:

```text
PRICE_ONLY_DELTA
ROUTINE_FUNDAMENTAL_DELTA
FULL_REFRESH_REQUIRED
```

For routine fundamental deltas, the analyst remains authoritative for material block impact.

Automation may:

- suggest dependencies;
- identify likely affected downstream blocks;
- enforce already-declared hard dependencies.

Automation may not:

- decide substantive analytical impact as a new deterministic rule;
- silently preserve a stale block solely because a dependency graph omitted it.

## 17. Contract pinning

Every live VNext run must pin exact immutable contract identities before execution.

A Module Contract used in a run must resolve to:

```text
logical name
version
content SHA256
durable immutable locator
```

No module may silently upgrade mid-run.

Material contract change requires the controlled successor/migration policy.

## 18. Security boundary

Browser clients do not mutate Registry state directly.

Target write path:

```text
AUTHORIZED SERVER ACTION
-> VALIDATE ENVIRONMENT
-> VALIDATE CONTRACT / STATE
-> CONTROLLED RPC
-> APPEND EVENT
```

Service-role capability is not itself authorization to violate contracts.

## 19. Observability boundary

Operational telemetry may capture:

- run/stage/module identifiers;
- transition type;
- duration;
- retries;
- blocker codes;
- error classes;
- artifact counts;
- hash-validation results;
- environment identity.

Telemetry must not become an alternative analytical source of truth.

No secret may enter logs or artifact metadata.

## 20. Gate 7 acceptance matrix

```text
G7-01 architecture layers explicit
G7-02 authority hierarchy explicit
G7-03 stages remain RESEARCH / DEEP_DIVE / INTEGRATION
G7-04 Module Contract fields machine representable
G7-05 module kinds have bounded authority
G7-06 Deep Dive block lifecycle preserves frozen execution vocabulary
G7-07 module transitions deterministic and fail closed
G7-08 canonical Deep Dive dependency order preserved
G7-09 safe parallelization does not bypass hard dependencies
G7-10 UNKNOWN / MISSING / N/A distinctions preserved
G7-11 no assumption fabrication to finish work
G7-12 Research insufficiency blocks downstream admission
G7-13 recovery does not depend on conversation memory
G7-14 exact artifact identity required downstream
G7-15 CHECKPOINT cannot become downstream admission authority
G7-16 FINAL required for stage handoff
G7-17 Integration cannot repair analytical defects
G7-18 refresh dependency automation remains advisory where analyst judgment is authoritative
G7-19 shadow project is the only allowed VNext Supabase target
G7-20 production project is explicitly forbidden
G7-21 publication disabled in VNext shadow
G7-22 no production current snapshot mutation
G7-23 module contract pinning required
G7-24 browser cannot directly mutate Registry
G7-25 architecture introduces no scoring / valuation / Certification methodology change
```

Gate 7 passes only when:

```text
PROSE SPEC
+ MACHINE-READABLE MODULE CONTRACT
+ RUNTIME VALIDATION
+ DETERMINISTIC TESTS
+ FULL VNEXT CI PASS
```

all agree, with no production mutation.


---

## 21. Gate 7 implementation authority

The prose architecture is paired with the following machine-enforced artifacts:

```text
schemas/vnext/orotitan-vnext-module-contract.schema.v0.1.json
  -> Module Contract structural authority

runtime/vnext/module-contract.ts
  -> runtime schema + cross-field invariant validation

tests/vnext-module-contract.test.ts
  -> deterministic architecture boundary tests

runtime/vnext/environment.ts
  -> hard VNEXT_SHADOW environment boundary

runtime/vnext/state-machine.ts
  -> Registry run/stage transition authority

runtime/vnext/state-model.ts
  -> orthogonal Gate 6 state-vector authority
```

Runtime Module Contract validation additionally enforces:

```text
SELF DEPENDENCY                         = FORBIDDEN
DEEP_DIVE_BLOCK_LIFECYCLE              = DEEP_DIVE ONLY
DEEP_DIVE EXECUTION VOCABULARY         = EXACT FROZEN VOCABULARY
AUTHORITATIVE STAGE OUTPUT             = PERSISTENCE + HASH REQUIRED
DETERMINISTIC MODULE JUDGMENT          = FORBIDDEN
DETERMINISTIC MODULE AUTHORITY FLAG    = REQUIRED
RESEARCH MODULE KIND                   = RESEARCH STAGE
ANALYTICAL / CERTIFICATION MODULE KIND = DEEP_DIVE STAGE
INTEGRATION MODULE KIND                = INTEGRATION STAGE
PRODUCTION ENVIRONMENT / WRITE         = SCHEMA-REJECTED
```

These checks constrain execution architecture only. They do not create analytical scoring rules.

## 22. Gate 7 freeze record

```text
GATE                              = 7
RESULT                            = PASS
PROSE_SPEC                        = PASS
MODULE_CONTRACT_JSON_SCHEMA       = PASS
RUNTIME_MODULE_VALIDATION         = PASS
DETERMINISTIC_MODULE_TESTS        = PASS
GATE_6_DEPENDENCY                 = FROZEN / PASS
ENVIRONMENT_FIREWALL              = PASS
UNKNOWN_DISCIPLINE                = PRESERVED
RECOVERY_ARCHITECTURE             = DEFINED
PUBLICATION_IN_SHADOW             = FORBIDDEN
CI_VALIDATION_HEAD                = c6e7e3185f7e9bb5db44757ab76e6d382664df0f
CI_RUN                            = 29
PRODUCTION_MUTATION               = NONE
SHADOW_DB_MUTATION                = NONE
ANALYTICAL_METHODOLOGY_CHANGE     = NONE
```

V0.1 is now the conceptual VNext architecture authority for subsequent implementation gates. Changes require an explicit new version; silent semantic mutation is forbidden.
