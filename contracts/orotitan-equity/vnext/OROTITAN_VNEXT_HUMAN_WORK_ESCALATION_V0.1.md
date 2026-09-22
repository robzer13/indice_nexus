# OROTITAN_VNEXT_HUMAN_WORK_ESCALATION_V0.1

**Project:** OroTitan Equity Research  
**Status:** FROZEN — GATE 14 PASS  
**Methodology change:** NO  
**Depends on:** Gates 7 through 12  
**Gate 13 dependency:** NONE for deterministic escalation mechanics  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

Gate 14 defines the boundary between the autonomous VNext runtime and exceptional human-assisted work.

ChatGPT Work is not a runtime API dependency.

It is a human-opened escalation environment used only when a case requires exceptional research, multi-tool inspection, inaccessible evidence handling, or unresolved judgment.

Core rule:

```text
AUTONOMOUS RUNTIME
-> DURABLE CHECKPOINT
-> PAUSE
-> HUMAN ESCALATION PACKAGE
-> HUMAN OPENS WORK
-> RESPONSE PERSISTED / ACCEPTED
-> EXACT RESUME CHECK
```

Stopping, closing, or abandoning Work must never destroy run state.

## 1. No runtime Work API

VNext runtime may not call ChatGPT Work as a hidden service.

The Work boundary is:

```text
invocation_mode = HUMAN_OPENED_CHATGPT_WORK
runtime_api_invocation_allowed = false
```

Work therefore cannot become:

- an orchestration dependency;
- a state-machine authority;
- a Registry writer;
- a publication authority;
- a hidden fallback model.

## 2. Escalation reasons

V0.1 bounded reasons:

```text
ANALYTICAL_AMBIGUITY
EVIDENCE_ACCESS_REQUIRED
MATERIAL_CONFLICT
MODEL_DISAGREEMENT
IDENTITY_AMBIGUITY
DATA_LICENSE_EXCEPTION
DATA_RESIDENCY_EXCEPTION
BUDGET_OVERRIDE_REQUIRED
UNCLASSIFIED_HUMAN_JUDGMENT
```

These are routing/escalation reasons, not analytical verdicts.

## 3. Durable checkpoint before handoff

Implementation:

```text
runtime/vnext/human-escalation.ts
```

The execution order is mandatory:

```text
READ CURRENT CONTEXT
-> VALIDATE EXACT RUN / STAGE / STATE_VERSION
-> ATOMIC CHECKPOINT + PAUSE
-> FRESH REREAD
-> VERIFY CHECKPOINT POINTER
-> VERIFY STATE VERSION INCREMENTS
-> VERIFY NO DATA_CUTOFF / CONTRACT / REVISION DRIFT
-> BUILD ESCALATION PACKAGE
-> PERSIST PACKAGE
-> STOP
```

The external human/Work activity begins only after this sequence.

## 4. Interruption safety

Once Gate 14 hands off:

```text
run_status       = PAUSED
stage_lifecycle  = PAUSED
active_manifest  = CHECKPOINT
```

If Work is closed, times out, loses browser state, is abandoned, or is resumed later, the persisted OroTitan run remains recoverable from the exact CHECKPOINT.

No conversational context is required for recovery.

## 5. Escalation package

The package contains:

```text
schema_version
escalation_id
RUN_ID
stage
reason
requested_human_action
checkpoint artifact ID/version/SHA256
DATA_CUTOFF
contract_set_sha256
stage_revision
run_state_version
stage_state_version
exact artifact references
blocker codes
Work authority boundary
resume requirements
```

Artifact references are exact and hashed.

Human summaries or conversational recollection cannot substitute for these references.

## 6. Work authority boundary

The generated package explicitly fixes:

```text
production_mutation_allowed = false
registry_mutation_allowed   = false
publication_allowed         = false
runtime_api_invocation      = false
```

Work may research, inspect, compare, explain, or produce a proposed resolution.

It cannot directly mutate authoritative OroTitan state.

## 7. Resume contract

A resume requires a persisted human-escalation response artifact.

Minimum controls:

```text
response artifact ID
response version
response SHA256
human acceptance = true
same RUN_ID
same stage
same active CHECKPOINT
exact run state_version
exact stage state_version
```

Any mismatch fails closed.

A Work answer pasted into conversation without persistence and acceptance is not resume authority.

## 8. State-model preservation

Gate 14 introduces no new top-level RunStatus or StageLifecycle values.

It uses the already frozen states:

```text
RunStatus       = PAUSED
StageLifecycle  = PAUSED
ManifestKind    = CHECKPOINT
```

WORK_ESCALATION_REQUIRED may exist as an event/reason or UI presentation code, but it is not a new state-machine value.

This avoids semantic drift in Gate 6 / Gate 7 frozen vocabularies.

## 9. Human-required examples

Typical cases:

```text
critical inaccessible evidence
material unresolved contradiction
issuer/security identity ambiguity
cross-model material disagreement
data-license / residency exception
hard budget override
exceptional analytical ambiguity
```

Normal provider errors remain under Gate 11 recovery classification and must not be escalated to Work merely because automation failed once.

## 10. Deterministic tests

Implementation:

```text
tests/vnext-human-escalation.test.ts
```

Covered cases:

```text
valid escalation checkpoints before handoff       PASS
run and stage become PAUSED                        PASS
CHECKPOINT becomes active                          PASS
handoff uses exact artifact hashes                 PASS
stale state_version                                REJECT
terminal run                                       REJECT
completed stage                                    REJECT
DATA_CUTOFF drift during pause                     DETECT
Work never returns                                 RUN REMAINS RECOVERABLE
accepted hashed response                           RESUME ELIGIBLE
wrong checkpoint                                   REJECT
unaccepted response                                REJECT
```

## 11. Gate 14 acceptance matrix

```text
G14-01 Work is not a runtime API dependency               PASS
G14-02 escalation reasons explicit                        PASS
G14-03 exact run/stage identity required                  PASS
G14-04 optimistic concurrency required                    PASS
G14-05 checkpoint created before external handoff         PASS
G14-06 run paused before external handoff                  PASS
G14-07 stage paused before external handoff                PASS
G14-08 active checkpoint independently reread             PASS
G14-09 DATA_CUTOFF drift detected                         PASS
G14-10 contract drift detected                            PASS
G14-11 stage revision drift detected                      PASS
G14-12 exact artifact hashes in handoff                   PASS
G14-13 Work has no production authority                   PASS
G14-14 Work has no Registry authority                     PASS
G14-15 Work has no publication authority                  PASS
G14-16 interruption leaves durable recoverable state      PASS
G14-17 resume requires persisted response artifact        PASS
G14-18 resume requires hash                               PASS
G14-19 resume requires explicit human acceptance          PASS
G14-20 resume requires same checkpoint                    PASS
G14-21 resume requires fresh state versions               PASS
G14-22 frozen state vocabulary unchanged                  PASS
G14-23 deterministic CI                                   PASS
```

## 12. Gate condition

Gate 14 passes when implementation, deterministic tests, full VNext CI, and protected PR merge all pass.

Gate 13 Azure quota availability is not required for this boundary because Gate 14 contains no model-provider execution.

## 13. Out of scope

Gate 14 does not:

- choose analytical models;
- change model routing policy;
- change analytical methodology;
- create a new state-machine vocabulary;
- publish;
- mutate production;
- bypass Gate 11 recovery;
- bypass Stage preflight or certification.


## 14. Freeze record

```text
GATE                              = 14
RESULT                            = PASS
WORK_RUNTIME_API_DEPENDENCY       = NONE
CHECKPOINT_BEFORE_HANDOFF         = PASS
PAUSE_BEFORE_HANDOFF              = PASS
INTERRUPTION_RECOVERABILITY       = PASS
EXACT_ARTIFACT_HASH_HANDOFF       = PASS
RESUME_CONCURRENCY_GUARDS         = PASS
PRODUCTION_AUTHORITY              = NONE
REGISTRY_AUTHORITY                = NONE
PUBLICATION_AUTHORITY             = NONE
STATE_VOCABULARY_CHANGE           = NONE
DETERMINISTIC_VNEXT_CI            = PASS
SCREENER_CI                       = PASS
PRODUCTION_MUTATION               = NONE
SHADOW_DB_MUTATION                = NONE
ANALYTICAL_METHODOLOGY_CHANGE     = NONE
```

V0.1 is frozen as the Gate 14 human-escalation boundary. Any semantic change requires an explicit new version.
