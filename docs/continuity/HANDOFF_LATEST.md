# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-037`

## Ministral 3 3B context4096 result

Status:

`PASS_LOAD_ONLY_MEASURED`

Exact model:
- `ministral-3:3b-instruct-2512-q4_K_M`;
- digest `f04aa1c738f64e13c625b82ae92504fc0260fa6723b509ed1ece0fa188179b1d`.

Measured:
- free RAM before: 1.14 GiB;
- free RAM loaded: 0.35 GiB;
- free RAM after unload: 1.85 GiB;
- VRAM used loaded: 2397 MiB;
- VRAM free loaded: 1566 MiB;
- processor split: `48%/52% CPU/GPU`;
- Ollama resident size: 3.1 GB;
- explicit unload: complete;
- semantic inference: none.

Interpretation:

`PASS_WITH_CRITICAL_RAM_PRESSURE`

This proves loadability only. It does not prove inference fit or production fit.

Relative to Qwen3 8B context4096, Ministral retains 0.14 GiB more free RAM and 13 percentage points more GPU residency, but has 118 MiB less VRAM headroom. Relative to Granite context4096, it retains 0.71 GiB less free RAM and 33 percentage points less GPU residency.

## Ministral 3 3B context8192 diagnostic

Authorization:

`G18-PHASEC-MINISTRAL3-3B-CONTEXT8192-LOAD-SMOKE-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-ministral3-3b-context8192-load-smoke.ts`

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
EXECUTE_MINISTRAL3_3B_CONTEXT8192_LOAD_ONLY_MEMORY_PREFLIGHT
```
