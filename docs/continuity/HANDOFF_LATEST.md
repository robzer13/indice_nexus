# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-POST-C7-20261002-073`

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

Data Contracts V2, Process Engine V2 and the ChatGPT ↔ Supabase Controlled Operation Contracts are frozen. The narrow bridge implementation remains the next required layer before a company vertical slice can be authorized.

## Active execution action

`IMPLEMENT_CHATGPT_SUPABASE_BRIDGE`

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


---

## Process Engine V2 — review + freeze (2026-10-02)

Implementation and review lineage:
- PR #338 merged into `vnext` at `180d817bcd522c9d1d528ce5e2521e48aadc9cce`;
- implementation head = `c5af5f982fec2e31273e8f98c14c9d5c1c945e07`;
- VNext CI #804 = PASS;
- Screener CI #662 = PASS;
- blocking Process Engine review findings = addressed / resolved;
- no methodology, scoring, valuation, production or Supabase-schema mutation.

Frozen authority:

`OROTITAN_PROCESS_ENGINE_V2_FREEZE_V1.0`

The freeze preserves:
- analytical execution status as `INSUFFICIENT | IN_PROGRESS | PROVISIONALLY_STABLE | LOCKED`;
- process freshness as the orthogonal `CURRENT | REOPENED | STALE` domain;
- dependency-cone reopen semantics;
- material-change revalidation;
- required overlay/dependency validation;
- deterministic next-block/action routing;
- refresh routing;
- SAVE disposition intent without persistence;
- execution fingerprint and bounded retry guard.

The Process Engine remains pure and does not own Supabase RPC invocation, artifact-byte persistence, registry transaction execution or publication.

Vertical slice:

`READY = NO`

Exact next action:

`DESIGN_CHATGPT_SUPABASE_CONTROLLED_OPERATION_CONTRACTS`


---

## ChatGPT ↔ Supabase Controlled Operation Contracts — review + freeze (2026-10-02)

Design and review lineage:
- PR #340 merged into `vnext` at `27d28d298de31c686a15aed74085d9531df565cb`;
- reviewed head = `95b47ceeeaceed92dc7004553f3207415d55cdfc`;
- VNext CI #809 = PASS;
- Screener CI #664 = PASS;
- review findings for current-stage coupling, BLOCK/CHECKPOINT lifecycle coupling and replay receipt consistency = FIXED / RESOLVED;
- no methodology, scoring, valuation, production or Supabase-schema mutation.

Frozen authority:

`OROTITAN_CHATGPT_SUPABASE_CONTROLLED_OPERATION_CONTRACTS_FREEZE_V1.0`

The freeze preserves:
- LOAD as mutation-free;
- exact run/stage optimistic concurrency state;
- exact artifact ID/version/hash/authority resolution;
- CHECKPOINT/BLOCK/FINALIZE mapping from Process Engine intent;
- stage-level REOPEN through the guarded Registry path;
- idempotency key + request fingerprint requirements;
- verified artifact persistence before Registry mutation;
- mandatory post-write durable-state verification;
- fail-closed stale-state handling;
- publication firewall with `publish_authorized = false`;
- no arbitrary SQL or arbitrary RPC surface.

Vertical slice:

`READY = NO`

Exact next action:

`IMPLEMENT_CHATGPT_SUPABASE_BRIDGE`
