# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-006`

## Current state

Gate 18 remains `IN_PROGRESS_NOT_FROZEN`.

The v1.1 validation contract is implemented and CI-verified.

The user explicitly authorized exactly one local Phi-4 / Constellation Software C4 inference under v1.1.

## Authorized cell

Company: `Constellation Software`  
Archetype: `SERIAL_ACQUIRER`  
Model: `phi4-mini:3.8b-q4_K_M`  
Validation: `GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1`

Frozen parameters:

- context: 16384
- max output: 1024
- temperature: 0
- client timeout: 480000 ms
- transport: NODE_HTTP_REQUEST_LOOPBACK
- keep_alive: 0s
- exact authorized run count: 1
- automatic retry: forbidden

Runner:

`scripts/vnext-gate18-phase-c-c4-constellation-phi4-mini-v1-1-context16384-output1024-timeout480-loopback-guarded.ts`

Authorization artifact:

`calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_PHI4_MINI_V1_1_CONTEXT16384_OUTPUT1024_TIMEOUT480_LOOPBACK_AUTH_001.json`

Authorization state:

`AUTHORIZED_SINGLE_LOCAL_INFERENCE`

## Preserved boundaries

This authorization covers one local inference only.

It does not authorize:

- a second attempt or automatic retry;
- parameter changes;
- model switch;
- Qwen3.5 download;
- production mutation;
- publication;
- historical v1.0 result mutation.

## Exact next action

```text
EXECUTE_EXACTLY_ONE_LOCAL_PHI4_CONSTELLATION_V1_1_INFERENCE
```

The execution must occur on the user's machine because the runner requires local Ollama and the local private repository.
