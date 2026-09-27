# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-018`

## Standing authority

`OROTITAN-STANDING-TECHNICAL-AUTH-002` remains ACTIVE for zero-cost technical execution. Nonzero external monetary cost still requires explicit user authorization.

## Gemma 3 4B pinned download

The authorized local download completed successfully.

Exact local identity:
- model: `gemma3:4b-it-q4_K_M`;
- full digest: `a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`;
- size: 3,338,801,804 bytes;
- format: `gguf`;
- family: `gemma3`;
- Ollama parameter size: `4.3B`;
- quantization: `Q4_K_M`.

Verification:
- exact tag present;
- expected public digest prefix matched;
- exact full digest captured;
- API show reachable;
- expected quantization matched;
- no load smoke during download;
- no prompt;
- no inference;
- no automatic retry;
- no paid execution;
- no production mutation.

Public result artifact:

`calibration/vnext/OROTITAN_GATE18_PHASE_C_GEMMA3_4B_DOWNLOAD_RESULT_001.json`

## Context4096 load-only authorization

A separate zero-cost load-only preflight is authorized under the standing technical authorization.

Authorization:

`G18-PHASEC-GEMMA3-4B-LOAD-SMOKE-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-gemma3-4b-load-smoke.ts`

Guards:
- exact full digest required;
- context exactly 4096;
- one load-only run;
- no prompt;
- no semantic inference;
- explicit unload after measurement;
- no automatic retry;
- no context change;
- no model switch;
- no production mutation.

## Exact next action

```text
EXECUTE_GEMMA3_4B_CONTEXT4096_LOAD_ONLY_MEMORY_PREFLIGHT
```

After the measured result is returned, decide whether an intermediate context8192 load-only preflight is justified before any larger-context load or inference.
