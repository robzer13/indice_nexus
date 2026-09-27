# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-031`

## Qwen3 8B context4096 result

Status:

`PASS_LOAD_ONLY_MEASURED`

Exact model:
- `qwen3:8b-q4_K_M`;
- digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`.

Measured:
- free RAM before: 1.38 GiB;
- free RAM loaded: 0.21 GiB;
- free RAM after unload: 3.75 GiB;
- VRAM used loaded: 2279 MiB;
- VRAM free loaded: 1684 MiB;
- Ollama processor split: `61%/39% CPU/GPU`;
- Ollama reported resident size: 6.0 GB;
- explicit unload: complete;
- semantic inference: none.

Interpretation:

`PASS_WITH_CRITICAL_RAM_PRESSURE`

This proves only that the model can be loaded at context4096. It does not prove inference fit or production fit.

Relative to Granite context4096:
- Qwen3 8B retains 0.85 GiB less free system RAM;
- VRAM headroom is similar (+34 MiB for Qwen3 8B);
- GPU residency drops from 85% to 39%, so most of the larger model is CPU/RAM-offloaded.

## Qwen3 8B context8192 diagnostic

Authorization:

`G18-PHASEC-QWEN3-8B-CONTEXT8192-LOAD-SMOKE-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-qwen3-8b-context8192-load-smoke.ts`

Purpose:
Empirically determine whether a context large enough for the target C4 prompt envelope is loadable before any inference decision.

Guards:
- exact digest required;
- context exactly 8192;
- one load-only run;
- no prompt;
- no semantic inference;
- explicit unload;
- no automatic retry;
- no model switch;
- no context growth beyond 8192;
- context16384 is not pre-authorized;
- production mutation forbidden.

## Exact next action

```text
EXECUTE_QWEN3_8B_CONTEXT8192_LOAD_ONLY_MEMORY_PREFLIGHT
```
