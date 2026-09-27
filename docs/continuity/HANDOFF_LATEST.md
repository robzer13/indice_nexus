# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-026`

## Granite 4 3B context4096 result

Status:

`PASS_LOAD_ONLY_MEASURED`

Exact identity:
- model: `granite4:3b`;
- digest: `89962fcc75239ac434cdebceb6b7e0669397f92eaef9c487774b718bc36a3e5f`;
- context: 4096.

Measured:
- free RAM before: 1.48 GiB;
- free RAM loaded: 1.06 GiB;
- free RAM after unload: 1.71 GiB;
- VRAM used loaded: 2313 MiB;
- VRAM free loaded: 1650 MiB;
- Ollama processor split: `15%/85% CPU/GPU`;
- Ollama reported resident size: 2.7 GB;
- explicit unload: complete;
- inference: none.

Relative to Gemma 3 at context4096, Granite retains +0.48 GiB more free RAM, +136 MiB more free VRAM, and substantially higher GPU residency.

Conclusion:
context4096 hardware fit passes with materially better headroom than recent candidates. One bounded context8192 load-only measurement is justified.

## Granite 4 context8192 authorization

Authorization:

`G18-PHASEC-GRANITE4-3B-CONTEXT8192-LOAD-SMOKE-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-granite4-3b-context8192-load-smoke.ts`

Guards:
- exact full digest required;
- context exactly 8192;
- one load-only run;
- no prompt;
- no semantic inference;
- explicit unload;
- no automatic retry;
- no context growth beyond 8192 in this run;
- no model switch;
- no production mutation.

## User-directed next candidate

Qwen3 8B remains explicitly queued immediately after the Granite sequence. Ministral 3 and Llama 3.2 remain deferred until after Qwen3 8B.

## Exact next action

```text
EXECUTE_GRANITE4_3B_CONTEXT8192_LOAD_ONLY_MEMORY_PREFLIGHT
```
