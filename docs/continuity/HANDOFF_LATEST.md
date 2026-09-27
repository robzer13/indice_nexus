# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-032`

## Qwen3 8B context8192 result

Status:

`PASS_LOAD_ONLY_MEASURED`

Measured:
- free RAM before: 1.20 GiB;
- free RAM loaded: 0.22 GiB;
- free RAM after unload: 3.07 GiB;
- VRAM used loaded: 2323 MiB;
- VRAM free loaded: 1640 MiB;
- processor split: `64%/36% CPU/GPU`;
- Ollama resident size: 6.6 GB;
- explicit unload: complete;
- semantic inference: none.

Interpretation:

`PASS_WITH_CRITICAL_RAM_PRESSURE`

Relative to context4096, loaded free RAM is effectively unchanged (+0.01 GiB), VRAM headroom is only 44 MiB lower, and GPU residency declines by 3 percentage points. This supports one final load-only diagnostic at context16384 for direct protocol comparability, but does not prove inference fit.

## Qwen3 8B context16384 final diagnostic

Authorization:

`G18-PHASEC-QWEN3-8B-CONTEXT16384-LOAD-SMOKE-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-qwen3-8b-context16384-load-smoke.ts`

Guards:
- exact digest required;
- context exactly 16384;
- one load-only run;
- no prompt;
- no semantic inference;
- explicit unload;
- no automatic retry;
- no model switch;
- no context growth beyond 16384;
- no production mutation.

After this measurement, explicitly decide between:
1. bounded comparable Constellation inference; or
2. hardware-stop disposition.

## Exact next action

```text
EXECUTE_QWEN3_8B_CONTEXT16384_LOAD_ONLY_MEMORY_PREFLIGHT
```
