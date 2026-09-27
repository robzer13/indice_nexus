# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-005`

## Current state

Gate 18 remains `IN_PROGRESS_NOT_FROZEN`.

The versioned v1.1 validation contract is implemented and CI-verified.

The post-v1.1 Phi-4 C4 resumption decision has been completed.

## Selected next discriminator

Company: `Constellation Software`  
Archetype: `SERIAL_ACQUIRER`  
Model: `phi4-mini:3.8b-q4_K_M`  
Validation: `GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1`

Selection basis:

- smallest remaining Phi-4 C4 workload after the already-run STMicro cell;
- closest workload size to STMicro, reducing runtime-size confounding;
- distinct archetype;
- absent from the six-case shadow replay, so it adds genuinely new model-behavior information;
- Qwen3 4B had engineering PASS / human PASS_WITH_CARRY on the same company, providing descriptive cross-candidate context.

This is not a model ranking or routing decision.

## Frozen planned cell

- context: 16384
- max output: 1024
- temperature: 0
- client timeout: 480000 ms
- transport: NODE_HTTP_REQUEST_LOOPBACK
- one run maximum
- automatic retry: forbidden
- raw output: preserved
- human adjudication: required

Runner:

`scripts/vnext-gate18-phase-c-c4-constellation-phi4-mini-v1-1-context16384-output1024-timeout480-loopback-guarded.ts`

Preparation:

`calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_PHI4_MINI_V1_1_PREP_001.json`

Authorization template:

`calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_PHI4_MINI_V1_1_CONTEXT16384_OUTPUT1024_TIMEOUT480_LOOPBACK_AUTH_001.json`

The template is intentionally `PREPARED_NOT_AUTHORIZED`; the runner requires `AUTHORIZED_SINGLE_LOCAL_INFERENCE` and therefore fails closed before Ollama inference.

## Preserved boundaries

No inference has been executed in this step.

No retry is authorized.

No model switch, Qwen3.5 download, production mutation, publication, or historical v1.0 result mutation is authorized.

## Exact next action

```text
AWAIT_EXPLICIT_SINGLE_RUN_PHI4_CONSTELLATION_V1_1_INFERENCE_AUTHORIZATION
```
