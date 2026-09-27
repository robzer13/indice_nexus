# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-013`

## Standing execution authority

`OROTITAN-STANDING-TECHNICAL-AUTH-002` remains ACTIVE.

Zero-cost OroTitan execution proceeds without repeated user confirmation. Explicit confirmation is required only before nonzero external monetary cost.

## Qwen3.5 target-context hardware

Context 16384 load-only: PASS.

Observed at context 16384:

- VRAM used: 2589 MiB
- VRAM headroom: 1374 MiB
- Ollama processor split: `56%/44% CPU/GPU`
- loaded free RAM: 0.51 GiB
- explicit unload: PASS

## First Qwen3.5 C4 Constellation run

Status:

`FAIL_RAW_JSON_INCOMPLETE_FORENSICS_REQUIRED`

Execution:

- company: Constellation Software
- model: `qwen3.5:4b-q4_K_M`
- context: 16384
- max output: 1024
- temperature: 0
- timeout: 600 seconds
- wall clock: 249631 ms
- done reason: `stop`
- prompt eval count: 3349
- eval count: 842
- output token margin: 182
- runtime error: none
- schema valid: false
- schema error: `Unexpected end of JSON input`
- v1.1 semantic validation: not reached

Private artifact:

`calibration/vnext/private-runs/2026-09-27T124959277Z__C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_QWEN3_5_4B_V1_1_CONTEXT16384_001.json`

Interpretation boundary:

- this is not a timeout;
- output budget exhaustion is not proven because `done_reason=stop` and 842 < 1024;
- raw structured-output termination failed;
- model capability failure is not yet concluded;
- no retry is authorized before read-only forensic diagnosis.

## Current forensic step

Authorization:

`G18-PHASEC-C4-CONSTELLATION-QWEN3_5-4B-RAW-JSON-FORENSIC-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-c4-constellation-qwen3-5-4b-raw-json-forensic.ts`

This forensic:

- reads the existing private artifact only;
- performs no model inference;
- makes no Ollama call;
- emits no raw generated content;
- measures structural progress and terminal JSON state;
- optionally probes structural closure on an in-memory copy only.

## Exact next action

```text
RUN_READ_ONLY_QWEN3_5_CONSTELLATION_RAW_JSON_TERMINATION_FORENSIC
```
