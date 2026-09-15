# OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V2 — FREEZE V2.0

**Status:** FROZEN DESIGN — V2.0  
**Depends on:** `OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0`  
**Methodology authority:** NONE  
**Production activation:** NOT AUTHORIZED BY THIS DOCUMENT ALONE

## 0. Role

Pilotage is permanent orchestration. It resolves run identity, verifies authoritative state, selects the legal next execution phase and emits an exact bootstrap. Pilotage never creates analytical truth, scores, valuation judgments or investment conclusions.

## 1. V2 execution phases

```text
RESEARCH
FUNDAMENTALS
VALUATION
CERTIFICATION_RECONCILIATION
INTEGRATION
```

Registry mapping:

```text
RESEARCH                      -> RESEARCH
FUNDAMENTALS                  -> DEEP_DIVE
VALUATION                     -> DEEP_DIVE
CERTIFICATION_RECONCILIATION  -> DEEP_DIVE
INTEGRATION                   -> INTEGRATION
```

## 2. Discussion names

```text
<COMPANY> — RESEARCH — <INITIAL|REFRESH> <YYYY-MM>
<COMPANY> — FUNDAMENTALS — <INITIAL|REFRESH> <YYYY-MM>
<COMPANY> — VALUATION — <INITIAL|REFRESH> <YYYY-MM>
<COMPANY> — CERTIFICATION — <INITIAL|REFRESH> <YYYY-MM>
<COMPANY> — INTEGRATION — <INITIAL|REFRESH> <YYYY-MM>
```

Names are UX labels only. `RUN_ID` is authoritative.

## 3. Bootstrap contract

Every bootstrap includes:

```text
COMPANY
RUN_ID
CANONICAL_MODE
RUN_TYPE
REGISTRY_STAGE
EXECUTION_PHASE
DATA_CUTOFF
EXPECTED_STAGE_CONTRACT
EXPECTED_STAGE_CONTRACT_VERSION
EXPECTED_INPUT_ARTIFACT_IDS / VERSIONS
BASELINE_SNAPSHOT_ID if applicable
HIGHER_AUTHORITY_PROCESS_VERSION
```

And exactly these invariants:

```text
DO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE.
LOAD AND VERIFY THE AUTHORITATIVE CONTRACT AND ARTIFACTS BEFORE WORK.
FAIL CLOSED ON VERSION / ID / STATUS / HASH MISMATCH.
```

## 4. Research -> Fundamentals admission

Pilotage must verify:

```text
RESEARCH_STAGE_STATUS = COMPLETE
READY_FOR_DEEP_DIVE = YES
active Research manifest = FINAL
required Research outputs = exact, available, hash-verified
no blocking Research condition
```

If valid, emit exact Fundamentals bootstrap. Otherwise emit exact blocker/resolution bootstrap.

## 5. Fundamentals -> Valuation admission

Pilotage must verify within the same `DEEP_DIVE` stage:

```text
DEEP_DIVE lifecycle = IN_PROGRESS
active Deep Dive manifest = CHECKPOINT
FUNDAMENTALS_LOCK = available + hash-verified
READY_FOR_VALUATION = YES in authoritative checkpoint payload
Fundamental Red Team = complete
no unresolved blocker preventing valuation
```

A Fundamentals CHECKPOINT never means `READY_FOR_INTEGRATION = YES`.

## 6. Valuation -> Certification admission

Pilotage must verify:

```text
DEEP_DIVE lifecycle = IN_PROGRESS
active Deep Dive manifest = CHECKPOINT
exact FUNDAMENTALS_LOCK version = resolved
VALUATION_LOCK = available + hash-verified
READY_FOR_CERTIFICATION = YES in authoritative checkpoint payload
no unresolved blocker preventing certification
```

A Valuation CHECKPOINT never admits Integration.

## 7. Certification -> Integration admission

Pilotage must verify:

```text
DEEP_DIVE_STAGE_STATUS = COMPLETE
READY_FOR_INTEGRATION = YES
active Deep Dive manifest = FINAL
required final Deep Dive artifacts = exact, available, hash-verified
no blocking Deep Dive execution defect
```

Only then may Pilotage emit Integration bootstrap.

## 8. Integration -> publication handoff

Pilotage must verify:

```text
INTEGRATION_STAGE_STATUS = COMPLETE
READY_TO_PUBLISH = YES
active Integration manifest = FINAL
candidate snapshot = exact + admitted
I2 = PASS
I3-B = PASS
```

Then the exact user command is:

```text
GO PUBLISH <COMPANY>
```

Pilotage never treats analytical `GO <COMPANY>` as publication authorization.

## 9. Mandatory final prompt rule

Each successful execution discussion ends with exactly one prompt for the next phase, and nothing follows it.

Each blocked discussion ends with exactly one resolution/Pilotage prompt, and nothing follows it.

The prompt may be copied by the user or reconstructed automatically by Pilotage. Both paths must resolve to the same `RUN_ID`, exact phase and exact artifact versions.

## 10. Auto-limited reopen routing

A downstream contradiction may trigger automatic limited reopen only when authoritative output identifies exact affected scope and reason.

```text
Valuation -> Fundamentals reopen
Certification -> Valuation reopen
Certification -> Fundamentals reopen
```

Pilotage must preserve prior artifact history, invalidate dependent eligibility and route only the affected scope. Ambiguous scope -> BLOCKED -> Pilotage review.

## 11. Refresh routing

Preserve classifier:

```text
PRICE_ONLY_DELTA
ROUTINE_FUNDAMENTAL_DELTA
FULL_REFRESH_REQUIRED
```

`PRICE_ONLY_DELTA` may legally skip a new Fundamentals discussion only when prior certified fundamental conclusions are explicitly revalidated for the successor run and no material fundamental delta exists.

## 12. V1 grandfathering

V1 runs continue under their immutable V1 pins. V2 must never be mixed into an existing V1 run. Material contract migration requires a successor run.

## 13. Fail-closed rules

Pilotage blocks on any identity, cutoff, contract, hash, artifact availability, manifest kind, gate-state or lineage mismatch. A human summary never satisfies admission.
