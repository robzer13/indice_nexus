# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-036`

## Ministral 3 3B pinned download

Status:

`PASS_PINNED_DOWNLOAD_ONLY`

Exact local identity:
- model: `ministral-3:3b-instruct-2512-q4_K_M`;
- digest: `f04aa1c738f64e13c625b82ae92504fc0260fa6723b509ed1ece0fa188179b1d`;
- size: 2,953,840,808 bytes;
- format: `gguf`;
- family: `mistral3`;
- Ollama parameter size: `3.8B`;
- public registry parameter size: `3.85B`;
- quantization: `Q4_K_M`.

No load, prompt, inference, retry, model switch, paid execution, or production mutation occurred during download.

## Context4096 load-only

Authorization:

`G18-PHASEC-MINISTRAL3-3B-CONTEXT4096-LOAD-SMOKE-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-ministral3-3b-context4096-load-smoke.ts`

Guards:
- exact full digest;
- context exactly 4096;
- one load-only run;
- no prompt;
- no inference;
- explicit unload;
- no retry;
- no context change;
- no model switch.

## Exact next action

```text
EXECUTE_MINISTRAL3_3B_CONTEXT4096_LOAD_ONLY_MEMORY_PREFLIGHT
```
