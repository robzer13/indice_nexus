# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-POST-C7-20261002-071`

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

`DESIGN_PROCESS_ENGINE_V2`

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

`DESIGN_PROCESS_ENGINE_V2`


---

## Data Contracts V2 — global review + freeze (2026-10-02)

The post-#334 global review found a material authority conflict between:
- the earlier consolidated post-C7 candidate; and
- the split schema package added by #333/#334.

Resolved on PR #336:
- split package is the single current authority candidate;
- prior consolidated package is historical only;
- frozen canonical `epistemic_type` is preserved;
- ChatGPT protocol `working_claim_type` is separate;
- frozen analytical block execution status remains `INSUFFICIENT | IN_PROGRESS | PROVISIONALLY_STABLE | LOCKED`;
- Process Engine control/reopen state must remain a separate layer;
- block references are bound to canonical `blockCode`.

Review result:
```text
VNext CI = PASS
Screener CI = PASS
Codex P1 = FIXED / RESOLVED
```

Freeze:
`OROTITAN_ANALYTICAL_DATA_CONTRACTS_V2_FREEZE_V1.0`

Vertical slice:
`READY = NO`

Exact next action:
`DESIGN_PROCESS_ENGINE_V2`
