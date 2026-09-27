# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-047`

## Granite 4.1 3B — first Constellation C4 engineering result

Status: `ENGINEERING_PASS_HUMAN_ADJUDICATION_REQUIRED`

Observed:
- wall clock: 72819 ms;
- done reason: stop;
- prompt eval count: 3145;
- eval count: 435 / 1024;
- output token margin: 589;
- runtime error: none;
- schema valid: true;
- semantic valid: true;
- V1.1 raw presentation compliant: true;
- normalized path count: 0;
- substantive status: PASS.

The single C4 authorization is consumed.

No retry or new Granite 4.1 inference is authorized.

## Current required step

`GENERATE_AND_REVIEW_PRIVATE_GRANITE4_1_HUMAN_ADJUDICATION_BUNDLE`

Source private run:

`calibration/vnext/private-runs/2026-09-27T214331230Z__C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_GRANITE4_1_3B_V1_1_CONTEXT16384_001.json`

Bundle builder:

`scripts/vnext-gate18-phase-c-granite4-1-3b-human-adjudication-bundle.ts`

The builder performs no model inference. It verifies the existing private run, reconstructs the exact pinned Constellation packet from the private repository, and emits a private human-adjudication bundle.

The private run, packet and bundle must not be committed or published.

Engineering PASS does not establish human-quality PASS, model ranking, routing, or production admission.

## Exact next action

`GENERATE_AND_REVIEW_PRIVATE_GRANITE4_1_HUMAN_ADJUDICATION_BUNDLE`
