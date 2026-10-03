# OroTitan Pilotage — Anti-Loop / Lossless Resume V1

Status: implementation note. This document is not a methodology authority and does not modify any frozen analytical or stage contract.

## Purpose

Prevent two orchestration failure modes:

1. **Looping:** the same authoritative run/stage state is dispatched to the same Pilotage operation repeatedly without a durable state/input delta.
2. **Context loss:** a new ChatGPT discussion reconstructs execution state from conversational memory instead of durable OroTitan state.

The implementation preserves the authority split:

- ChatGPT performs analysis and produces work.
- Supabase stores operational run/stage state and the continuation-attempt ledger.
- private artifact storage stores immutable analytical artifacts.
- Vercel hosts the runtime.
- OroTitan Pilotage routes, verifies, prepares and hands off.
- Chat memory is never an authority.

## Lossless resume envelope

A resume envelope is rebuilt from a fresh controlled `LOAD_RESULT`. It contains:

- exact `RUN_ID`;
- run and stage `state_version`;
- current stage and stage revision;
- lifecycle and handoff gate;
- frozen Contract Set SHA-256;
- exact active Stage Manifest ref when present;
- exact Process Engine State ref when present;
- exact `artifact_index`;
- exact context tiers L0-L3;
- authoritative blockers;
- deterministic state fingerprint;
- one derived `EXACT_NEXT_ACTION`.

Every artifact ref used by the resume envelope must be backed exactly by `LOAD_RESULT.artifact_index` with the same:

- `artifact_id`;
- `version`;
- `content_sha256`;
- required authority class.

Missing, partial, conflicting or reconstructed refs fail closed.

## State fingerprint

The state fingerprint is SHA-256 over canonicalized durable state only.

It includes run/stage versions, Contract Set identity, stage state, exact artifact identities, blockers and context plan.

It deliberately excludes:

- conversational wording;
- ChatGPT session identifiers;
- timestamps of the request;
- UI state;
- non-authoritative human summaries.

Equivalent durable states therefore produce the same fingerprint even when blocker/context list order differs.

## Continuation routing

The guard derives one next action:

| Durable state | Exact next action |
| --- | --- |
| BLOCKED or blockers present | `RESOLVE_BLOCKER` |
| PAUSED with durable anchor | `RESUME_STAGE` |
| PAUSED without durable anchor | `SAVE_DURABLE_CHECKPOINT` |
| IN_PROGRESS with durable anchor | `CONTINUE_STAGE` |
| IN_PROGRESS without durable anchor | `SAVE_DURABLE_CHECKPOINT` |
| COMPLETE + gate YES, Research/Deep Dive | `HANDOFF_NEXT_STAGE` |
| COMPLETE + gate YES, Integration | `AWAIT_EXPLICIT_GO_PUBLISH` |
| inconsistent/unsafe state | `FAIL_CLOSED` |

`GO_PUBLISH` is never automatically dispatched by this guard. Publication remains a separate explicitly authorized operation.

## Durable anti-loop ledger

`public.orotitan_pilotage_attempts` is operational state only.

The controlled RPC `register_orotitan_pilotage_attempt`:

1. locks the current run and stage;
2. verifies expected run/stage CAS versions;
3. rejects terminal or mismatched current-stage state;
4. recomputes the authoritative next action from durable database state;
5. rejects route mismatches;
6. keys replay detection to the durable `run_state_version + stage_state_version + requested_operation`;
7. stores the deterministic state fingerprint for that CAS state;
8. rejects fingerprint drift for the same CAS state;
9. returns `NO_PROGRESS_REPLAY` on the same durable state + operation thereafter.

A replay never authorizes an analytical or stage mutation.

Changing a chat session does not reset the ledger.

## Retry semantics

`FIRST_ATTEMPT` means the requested continuation is eligible to dispatch.

`NO_PROGRESS_REPLAY` means:

- same authoritative run and stage CAS versions;
- same requested operation;
- same deterministic state fingerprint;
- no durable delta proving progress.

A different fingerprint presented for the same run/stage CAS versions is rejected as fingerprint drift rather than treated as progress.

Required response:

`NO MUTATION -> RELOAD / PRODUCE DURABLE DELTA -> EXACT_NEXT_ACTION`

Existing block-level retry controls (`computeExecutionFingerprint`, `decideRetry`) remain unchanged and complementary.

## Security boundary

- RLS is enabled on the continuation ledger.
- no client RLS policy exists.
- no direct INSERT/UPDATE/DELETE/TRUNCATE grant exists for anon/authenticated/service_role.
- service_role receives SELECT only.
- writes occur only through the SECURITY DEFINER RPC.
- the RPC pins `search_path = pg_catalog, public`.
- the RPC does not mutate runs, stages, artifacts, snapshots or canonical publication pointers.

## Non-goals

This slice does not:

- produce Research, Deep Dive or Integration analysis;
- change any analytical score or valuation;
- change frozen contracts;
- authorize publication;
- create a Stage Manifest;
- replace artifact idempotency/CAS controls;
- make chat history authoritative.
