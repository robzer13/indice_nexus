# OROTITAN_VNEXT_RUN_CONTROLLER_V0.1

**Project:** OroTitan Equity Research  
**Status:** CANDIDATE FOR GATE 8  
**Methodology change:** NO  
**Depends on:** Gate 6 State Model + Gate 7 Architecture  
**AI dependency:** NONE

## 0. Purpose

Gate 8 implements the first deterministic VNext run controller.

Its scope is deliberately narrow:

```text
READ
-> DETERMINE LEGAL RUN TRANSITION
-> EXECUTE ONE DETERMINISTIC COMPARE-AND-SET MUTATION
-> REREAD
```

The controller does not perform Research, Deep Dive analysis, Integration, Certification, scoring, valuation, or publication.

It never infers state from chat memory.

## 1. Authority

Run-transition legality is delegated exclusively to:

```text
runtime/vnext/state-machine.ts
```

The controller may not add alternative transition paths.

The persistent store remains the eventual operational authority. The controller is an execution boundary around that store, not a replacement for Registry authority.

## 2. Input contract

Each transition request requires:

```text
runId
operationId
requestFingerprint
expectedStateVersion
targetStatus
```

No transition may be attempted without optimistic concurrency state.

## 3. Initial READ

The controller reads one context containing:

```text
current run status
current stateVersion
prior idempotency receipt for operationId, if any
```

If the run cannot be resolved exactly, the store must fail closed.

## 4. Idempotency

If `operationId` already exists:

- identical request fingerprint is treated as an idempotent replay;
- a different fingerprint fails with `VNEXT_IDEMPOTENCY_CONFLICT`;
- the operation cannot be replayed against another target status;
- no second mutation is executed.

## 5. Transition determination

For a new operation the controller verifies, in order:

```text
stored stateVersion == expectedStateVersion
shadow publication firewall
frozen run transition legality
```

A failure stops execution before mutation.

## 6. Deterministic execution

The store receives exactly one compare-and-set transition request containing:

```text
runId
operationId
requestFingerprint
expectedStateVersion
fromStatus
toStatus
```

The store implementation is responsible for atomic concurrency enforcement and durable receipt persistence.

Gate 8 does not yet bind this interface to production Supabase.

## 7. REREAD verification

After mutation the controller rereads persistent state and verifies:

```text
same RUN_ID
actual status == requested target
stateVersion == prior stateVersion + 1
idempotency receipt exists
receipt fingerprint matches
receipt before/after versions match observed state
```

Any mismatch fails closed.

## 8. Shadow publication firewall

In `VNEXT_SHADOW`:

```text
targetStatus = PUBLISHED
=> VNEXT_SHADOW_PUBLICATION_FORBIDDEN
```

A run may reach `READY_TO_PUBLISH` for pre-publication assurance, but Gate 8 cannot publish.

## 9. AI boundary

Gate 8 contains:

```text
LLM calls                 = 0
ChatGPT dependency        = 0
model provider dependency = 0
analytical judgment       = 0
```

A future model provider may produce analytical artifacts, but it will never receive authority to mutate the run state machine directly.

## 10. Durable orchestration provider

Vercel Workflows is selected as the preferred durable orchestration envelope for VNext.

Current Vercel documentation uses:

```text
'use workflow'
'use step'
package: workflow
```

The provider is intentionally kept outside transition authority.

Target architecture:

```text
Vercel Workflow
  -> durable scheduling / retry / suspension
  -> invokes deterministic Run Controller
  -> Run Controller validates frozen state machine
  -> Store performs atomic mutation
  -> Run Controller rereads and verifies
```

Gate 8 freezes the controller core first. The workflow adapter must call this same core and may not duplicate or override transition rules.

No unpinned workflow dependency is introduced in this gate.

## 11. Test store

Gate 8 validation uses an in-memory store implementing the same persistence interface.

The fake store provides:

- exact run lookup;
- optimistic compare-and-set;
- operation receipts;
- fingerprint conflict detection;
- controllable simulated race.

It contains no ChatGPT or model dependency.

## 12. Deterministic fake-run scenario

The required simulated lifecycle is:

```text
CREATED
-> ACTIVE
-> PAUSED
-> ACTIVE
-> BLOCKED
-> ACTIVE
-> READY_TO_PUBLISH
```

Expected:

```text
6 successful mutations
stateVersion 1 -> 7
PUBLISHED attempt in shadow -> rejected
```

Additional negative tests:

```text
illegal transition                  -> reject
stale expected stateVersion         -> reject
race after initial READ             -> CAS reject
same id + same fingerprint          -> idempotent replay
same id + different fingerprint     -> reject
same id + different target          -> reject
```

## 13. Implementation artifacts

```text
runtime/vnext/run-controller.ts
tests/vnext-run-controller.test.ts
runtime/vnext/state-machine.ts
contracts/orotitan-equity/vnext/OROTITAN_VNEXT_RUN_CONTROLLER_V0.1.md
```

## 14. Gate 8 acceptance matrix

```text
G8-01 READ before decision                         REQUIRED
G8-02 exact frozen transition matrix reused        REQUIRED
G8-03 optimistic stateVersion check                REQUIRED
G8-04 atomic store CAS boundary                    REQUIRED
G8-05 deterministic REREAD after mutation          REQUIRED
G8-06 post-mutation status verified                REQUIRED
G8-07 post-mutation stateVersion verified          REQUIRED
G8-08 durable operation receipt verified           REQUIRED
G8-09 idempotent replay                            REQUIRED
G8-10 conflicting fingerprint fails closed         REQUIRED
G8-11 concurrent race fails closed                 REQUIRED
G8-12 illegal transition fails before mutation     REQUIRED
G8-13 shadow publication hard-blocked              REQUIRED
G8-14 READY_TO_PUBLISH allowed in shadow           REQUIRED
G8-15 full fake lifecycle works without ChatGPT    REQUIRED
G8-16 no AI/model dependency                       REQUIRED
G8-17 Vercel Workflows owns durability, not law    REQUIRED
G8-18 no production mutation                       REQUIRED
G8-19 no shadow DB mutation                        REQUIRED
G8-20 full VNext CI                                REQUIRED
```

Gate 8 passes only when the fake run executes fully in CI and every negative-path test passes.

## 15. Out of scope

Gate 8 does not implement:

- PRE_STAGE_PREFLIGHT;
- POST_STAGE_CERTIFICATION;
- automated Recovery Engine;
- Supabase-backed run-controller store;
- model providers;
- analytical modules;
- publication;
- production promotion.

These remain later gates.
