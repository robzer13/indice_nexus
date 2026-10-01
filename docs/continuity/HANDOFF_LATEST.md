# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-POST-C7-20261001-071`

## Current state

ChatGPT operating protocol:

`OROTITAN_CHATGPT_OPERATING_PROTOCOL_V1_FREEZE_V1.0`

Status:

`FROZEN_V1_0`

Post-C7 architectural planning:

`CLOSED`

## Regression battery

Baseline protocol battery executed against:

- merged CI baseline;
- frozen 25-scenario acceptance matrix;
- live Supabase registry in read-only preflight mode.

Result:

`FOUNDATION_PASS_WITH_IMPLEMENTATION_GAPS`

Verified:

- VNext CI baseline = PASS;
- Screener CI baseline = PASS;
- RLS enabled on core registry tables;
- critical registry RPCs are service-role only;
- checkpoint/finalize/reopen version guards exist;
- idempotency controls exist;
- artifact resolver checks SHA-256 + authority class;
- publication remains separate from SAVE.

Security debt recorded:

- RLS-without-policy posture must be explicitly documented as service-only;
- 3 legacy functions have mutable `search_path` and must be hardened before final production hardening.

## Vertical slice

`VERTICAL_SLICE_READY = FALSE`

15 scenarios require implementation closure before vertical slice.

They cluster into:

1. Data Contracts V2
2. Process Engine V2
3. ChatGPT ↔ Supabase bridge

## Active execution action

`RUN_PROCESS_ENGINE_V2_CI_AND_REVIEW`

## Product invariant

OroTitan product surface / vitrine remains:

`FRENCH_FIRST_UI = TRUE`

Machine contracts and internal canonical vocabulary may remain English.

## Data Contracts V2 candidate

Drafted on branch `post-c7-data-contracts-v2-design-001`.

Current package:
- composable analytical JSON Schema;
- semantic validator;
- targeted regression tests;
- no live Supabase mutation.

Prior candidate action:

`RUN_DATA_CONTRACTS_V2_CI_AND_REVIEW` — completed on PR #328.

## Data Contracts V2 design review — CLOSED

Initial candidate CI already passed on PR #328.

Review found and patched on branch `post-c7-data-contracts-v2-review-fixes-001`:

- forward-estimate period cutoff bug;
- duplicate analytical-block ambiguity;
- root-source self/cycle integrity gap;
- missing evidence guards for COMPLETE blocks and SUPPORTED/MIXED causal links.

No Supabase or production mutation.

Exact next action:

`RUN_DATA_CONTRACTS_V2_REVIEW_FIX_CI`


## Data Contracts V2 review closure

Review result:

`PASS_STABLE_FOR_PROCESS_ENGINE`

Implementation lineage:
- initial design PR #328 merged at `cd4554ebac7695323f8ec71286652ab436dc6a13`;
- review-hardening PR #329 merged at `489c0e0f4bf41bfe75f25ee00549b927c1e99faa`;
- VNext CI = PASS;
- Screener CI = PASS.

The reviewed package is stable for Process Engine V2 design. It is not promoted as a new frozen analytical methodology and no production mutation occurred.

Exact next action:

`RUN_PROCESS_ENGINE_V2_CI_AND_REVIEW`


## Process Engine V2 design candidate

Active branch:

`post-c7-process-engine-v2-design-002`

The pre-existing unmerged branch `post-c7-process-engine-v2-design-001` is retained only as superseded draft history.

Candidate implements the deterministic process layer for:

- dependency-graph validation and limited downstream reopening;
- material revalidation completion gating;
- required sector-overlay validation;
- price-only / routine / full refresh routing;
- SAVE checkpoint/finalization eligibility;
- execution fingerprint / retry loop guard;
- blocker-aware next-block resolution.

Boundary remains strict:

```text
PROCESS ENGINE
≠ SUPABASE BRIDGE
≠ ANALYTICAL JUDGMENT
≠ PUBLICATION
```

No production mutation.

Exact next action:

`RUN_PROCESS_ENGINE_V2_CI_AND_REVIEW`


## Process Engine V2 review hardening

Pre-CI semantic review patched:

- authoritative dependency requirements to prevent under-specified reopen graphs;
- direct REOPENED vs downstream STALE distinction;
- SAVE pre-finalization decision no longer requires post-RPC registry reconciliation;
- lifecycle guards for NOT_STARTED / PAUSED / BLOCKED / COMPLETE;
- current executable block preserved ahead of unrelated later blockers;
- empty process state fails closed.

Exact action remains:

`RUN_PROCESS_ENGINE_V2_CI_AND_REVIEW`


Final Process Engine semantic hardening before review close:

- READY block cannot jump directly to terminal completion;
- persisted COMPLETE blocks are revalidated against completion / pinned method-plan conditions on resume;
- completion-audit, required-overlay and required-dependency inputs are explicitly Bridge-resolved pinned authority, not ad-hoc Process Engine judgments.
