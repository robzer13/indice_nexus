# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-010`

## Standing execution authority

`OROTITAN-STANDING-TECHNICAL-AUTH-002` remains ACTIVE.

Zero-cost OroTitan execution proceeds without repeated user confirmation. A new confirmation is required only before a nonzero external monetary cost.

## Qwen3.5 4B identity

- model: `qwen3.5:4b-q4_K_M`
- digest: `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`
- quantization: `Q4_K_M`
- download identity: PASS

## Context 4096 load-only result

Status: `PASS_LOAD_ONLY_MEASURED`

Observed:

- total RAM: 7.84 GiB
- free RAM before: 1.33 GiB
- free RAM loaded: 0.69 GiB
- free RAM after unload: 2.58 GiB
- VRAM used loaded: 2601 MiB
- VRAM headroom loaded: 1362 MiB
- Ollama loaded size: 3.7 GB
- processor split: `50%/50% CPU/GPU`
- prompt: none
- semantic inference: none
- unload: PASS
- full VRAM release: PASS

RAM free-memory snapshots are treated as noisy OS-level measurements because the post-unload value exceeded the pre-load baseline.

Compared with Qwen3 4B at context 4096, Qwen3.5 used 290 MiB more VRAM and shifted materially toward CPU offload.

## Interpretation

The context-4096 load fit is real, but headroom is tighter than the Qwen3 4B reference.

A direct jump to context 16384 is not the cheapest discriminating next step.

The bounded next step is an intermediate context-8192 load-only memory measurement.

## Current authorized step

`G18-PHASEC-QWEN3_5-4B-CONTEXT8192-LOAD-SMOKE-AUTH-001`

- context: 8192
- load only
- no prompt
- no semantic generation
- measure RAM/VRAM and Ollama processor split
- explicit unload
- one run
- zero cost

## Exact next action

```text
EXECUTE_QWEN3_5_4B_CONTEXT8192_LOAD_ONLY_MEMORY_PREFLIGHT
```
