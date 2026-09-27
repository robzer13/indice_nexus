# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-011`

## Standing execution authority

`OROTITAN-STANDING-TECHNICAL-AUTH-002` remains ACTIVE.

Zero-cost execution proceeds without repeated user confirmation. Explicit confirmation is required only before nonzero external monetary cost.

## Qwen3.5 4B identity

- model: `qwen3.5:4b-q4_K_M`
- digest: `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`
- quantization: `Q4_K_M`

## Context 4096 load-only result

- status: PASS
- VRAM used: 2601 MiB
- VRAM headroom: 1362 MiB
- Ollama size: 3.7 GB
- processor split: `50%/50% CPU/GPU`
- loaded free RAM: 0.69 GiB

## Context 8192 load-only result

- status: PASS
- VRAM used: 2525 MiB
- VRAM headroom: 1438 MiB
- Ollama size: 3.9 GB
- processor split: `54%/46% CPU/GPU`
- loaded free RAM: 0.56 GiB
- unload: PASS
- semantic inference: none

## Scaling interpretation

From 4096 to 8192:

- Ollama loaded size increased by about 0.2 GB;
- CPU offload increased from 50% to 54%;
- measured VRAM use decreased by 76 MiB;
- RAM pressure remained high.

This indicates that Ollama is absorbing context growth by moving more of the model/runtime burden to CPU/RAM rather than driving VRAM toward saturation.

The target C4 context is 16384, so the next cheapest discriminating step is one direct load-only measurement at 16384 before any inference.

## Current authorized step

`G18-PHASEC-QWEN3_5-4B-CONTEXT16384-LOAD-SMOKE-AUTH-001`

- context: 16384
- load only
- no prompt
- no semantic generation
- measure RAM/VRAM and Ollama processor split
- explicit unload
- one run
- zero cost

## Exact next action

```text
EXECUTE_QWEN3_5_4B_CONTEXT16384_LOAD_ONLY_MEMORY_PREFLIGHT
```
