# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-025`

## Granite 4 3B pinned download

Status:

`PASS_PINNED_DOWNLOAD_ONLY`

Exact local identity:
- model: `granite4:3b`;
- full digest: `89962fcc75239ac434cdebceb6b7e0669397f92eaef9c487774b718bc36a3e5f`;
- size: 2,099,521,385 bytes;
- format: `gguf`;
- family: `granite`;
- parameter size: `3.4B`;
- quantization: `Q4_K_M`.

No load smoke, prompt, inference, retry, model switch, paid execution, or production mutation occurred in the download run.

## Granite 4 context4096 load-only

Authorization:

`G18-PHASEC-GRANITE4-3B-LOAD-SMOKE-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-granite4-3b-load-smoke.ts`

Guards:
- exact full digest required;
- context exactly 4096;
- one load-only run;
- no prompt;
- no semantic inference;
- explicit unload;
- no automatic retry;
- no context change;
- no model switch;
- no production mutation.

## User-directed candidate sequence

The user explicitly requires Qwen3 8B to be tested after Granite.

Sequence now preserved:
1. current: `GRANITE4_3B_OLLAMA_Q4_K_M`;
2. next after Granite: `QWEN3_8B_LOCAL`.

Ministral 3 3B and Llama 3.2 3B are deferred until after the Qwen3 8B test.

Qwen3 8B will not be jumped directly into inference. Required sequence:
`pinned download -> context4096 load-only -> context8192 if supported -> context16384 if supported -> inference only if hardware qualification supports it`.

## Exact next action

```text
EXECUTE_GRANITE4_3B_CONTEXT4096_LOAD_ONLY_MEMORY_PREFLIGHT
```
