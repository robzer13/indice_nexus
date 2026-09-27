# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-012`

## Standing execution authority

`OROTITAN-STANDING-TECHNICAL-AUTH-002` remains ACTIVE.

Zero-cost OroTitan execution proceeds without repeated user confirmation. Explicit confirmation is required only before a nonzero external monetary cost.

## Qwen3.5 4B hardware qualification

Identity:

- model: `qwen3.5:4b-q4_K_M`
- digest: `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`
- quantization: `Q4_K_M`

Measured load-only results:

| Context | VRAM used | VRAM headroom | Ollama size | CPU/GPU | Free RAM loaded |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | 2601 MiB | 1362 MiB | 3.7 GB | 50%/50% | 0.69 GiB |
| 8192 | 2525 MiB | 1438 MiB | 3.9 GB | 54%/46% | 0.56 GiB |
| 16384 | 2589 MiB | 1374 MiB | 4.2 GB | 56%/44% | 0.51 GiB |

All three load-only tests passed and explicitly unloaded the model. No Qwen3.5 semantic inference has yet occurred.

Interpretation:

- target context 16384 is loadable;
- VRAM remains materially below saturation;
- context growth is primarily absorbed by increasing CPU/RAM offload;
- system RAM pressure is high;
- bounded inference fit is plausible but not yet proven.

## First Qwen3.5 C4 discriminator

Selected company: `Constellation Software`  
Archetype: `SERIAL_ACQUIRER`

Why this cell:

- Qwen3 4B: engineering PASS / human PASS_WITH_CARRY;
- Phi-4: engineering PASS under v1.1 / human CRITICAL_FAILURE;
- same pinned packet allows direct cross-candidate comparison of grounding and priority selection.

Frozen execution:

- model: `qwen3.5:4b-q4_K_M`
- context: 16384
- max output: 1024
- temperature: 0
- timeout: 600000 ms
- transport: `NODE_HTTP_REQUEST_LOOPBACK`
- generation prompt contract: v1.0 unchanged
- generation schema contract: v1.0 unchanged
- validation: `GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1`
- automatic retry: false
- human adjudication required

Authorization:

`G18-PHASEC-C4-CONSTELLATION-QWEN3_5-4B-V1_1-CONTEXT16384-OUTPUT1024-TIMEOUT600-LOOPBACK-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-c4-constellation-qwen3-5-4b-v1-1-context16384-output1024-timeout600-loopback-guarded.ts`

## Exact next action

```text
EXECUTE_FIRST_BOUNDED_QWEN3_5_CONSTELLATION_C4_INFERENCE_V1_1
```
