# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-016`

## Standing authority

`OROTITAN-STANDING-TECHNICAL-AUTH-002` remains ACTIVE. Zero-cost execution proceeds without reprompt. Any action with nonzero external monetary cost still requires explicit user authorization.

## Qwen3.5 Constellation same-cell retry

The historical first Qwen3.5 Constellation attempt remains FAIL and runtime-adapter confounded. It is not reclassified.

The separately authorized same-cell compatibility retry with `think:false` was executed locally and reached an engineering PASS.

Observed:
- model `qwen3.5:4b-q4_K_M`;
- exact pinned digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`;
- Constellation Software / SERIAL_ACQUIRER;
- context 16384;
- max output 1024;
- temperature 0;
- timeout 600 seconds;
- wall clock 283571 ms;
- done reason `stop`;
- prompt eval count 3351;
- eval count 953;
- output-token margin 71;
- runtime error none;
- schema valid;
- raw presentation compliant;
- safe normalization count 0;
- substantive deterministic status PASS;
- semantic valid;
- external model API cost USD 0.

Public result artifact:

`calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_QWEN3_5_4B_V1_1_THINK_FALSE_RESULT_001.json`

Private raw output remains local and is not published:

`calibration/vnext/private-runs/2026-09-27T143133027Z__C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_QWEN3_5_4B_V1_1_CONTEXT16384_THINKFALSE_001.json`

## Interpretation boundary

This result is `ENGINEERING_PASS_HUMAN_ADJUDICATION_PENDING` only.

It does not authorize:
- a model winner;
- model ranking;
- routing freeze;
- production mutation;
- publication of private source material;
- broader Qwen3.5 C4 expansion.

No automatic second retry is authorized.

## Exact next action

```text
COMPLETE_QWEN3_5_CONSTELLATION_HUMAN_QUALITY_ADJUDICATION
```

Human adjudication must review the private raw output as emitted against the pinned private evidence packet before deciding Qwen3.5 expansion or stop.
