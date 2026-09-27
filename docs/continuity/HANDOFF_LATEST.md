# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-034`

## Qwen3 8B Constellation C4

Engineering status:

`ENGINEERING_PASS_HUMAN_ADJUDICATION_REQUIRED`

Observed:
- model: `qwen3:8b-q4_K_M`;
- exact digest preserved;
- `think:false`;
- baseline free RAM: 3.59 GiB;
- minimum guard: 1.0 GiB;
- wall clock: 519551 ms;
- done reason: `stop`;
- prompt eval: 3356;
- eval: 860 / 1024;
- output margin: 164;
- runtime error: null;
- schema PASS;
- semantic PASS;
- raw presentation PASS;
- normalized paths: 0.

This is the strongest engineering result observed so far on the same Constellation C4 cell, but human quality is not yet adjudicated and production hardware fit remains unproven.

## Human adjudication next step

Public prep:
`G18-PHASEC-C4-CONSTELLATION-QWEN3-8B-HUMAN-ADJUDICATION-PREP-001`

Private bundle runner:
`scripts/vnext-gate18-phase-c-qwen3-8b-human-adjudication-bundle.ts`

The bundle reconstructs the exact pinned Constellation packet from the private repo and combines it with the original private model response. It writes only under `calibration/vnext/private-runs/`; it performs no inference and publishes no private source content.

Required checks include:
- E-045 RFP overstatement;
- E-029 acquisition-criteria universalization;
- E-063 false unresolved-retention question;
- duplicate recurring-revenue priority slots;
- omission of E-066/E-067;
- handling of C-005.

No further Qwen3 8B inference or retry is authorized.

## Exact next action

```text
GENERATE_PRIVATE_QWEN3_8B_HUMAN_ADJUDICATION_BUNDLE
```
