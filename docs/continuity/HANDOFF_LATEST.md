# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-027`

## Granite 4 3B context8192 result

Status:

`PASS_LOAD_ONLY_MEASURED`

Measured:
- free RAM before: 1.84 GiB;
- free RAM loaded: 0.94 GiB;
- free RAM after unload: 1.80 GiB;
- VRAM used loaded: 2355 MiB;
- VRAM free loaded: 1608 MiB;
- Ollama processor split: `22%/78% CPU/GPU`;
- Ollama reported resident size: 3.1 GB;
- explicit unload: complete;
- semantic inference: none.

Relative to Granite context4096:
- loaded free RAM decreased by 0.12 GiB;
- VRAM headroom decreased by 42 MiB;
- GPU residency remains high at 78%.

Relative to Gemma 3 at context8192:
- Granite retains +0.17 GiB more free RAM;
- Granite retains +68 MiB more free VRAM;
- Granite GPU residency is 78% vs 44%.

Conclusion:
Granite context8192 hardware fit passes with useful remaining headroom. One bounded context16384 load-only measurement is justified before any inference.

## Granite 4 context16384 authorization

Authorization:

`G18-PHASEC-GRANITE4-3B-CONTEXT16384-LOAD-SMOKE-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-granite4-3b-context16384-load-smoke.ts`

Guards:
- exact full digest required;
- context exactly 16384;
- one load-only run;
- no prompt;
- no semantic inference;
- explicit unload;
- no automatic retry;
- no context growth beyond 16384 in this run;
- no model switch;
- no production mutation.

## User-directed next candidate

Qwen3 8B remains explicitly queued immediately after the Granite sequence. Ministral 3 and Llama 3.2 remain deferred until after Qwen3 8B.

## Exact next action

```text
EXECUTE_GRANITE4_3B_CONTEXT16384_LOAD_ONLY_MEMORY_PREFLIGHT
```
