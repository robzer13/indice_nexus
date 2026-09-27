# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-030`

## Qwen3 8B pinned download

Status:

`PASS_PINNED_DOWNLOAD_ONLY`

Exact local identity:
- model: `qwen3:8b-q4_K_M`;
- full digest: `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`;
- size: 5,225,388,164 bytes;
- format: `gguf`;
- family: `qwen3`;
- parameter size: `8.2B`;
- quantization: `Q4_K_M`.

No load smoke, prompt, inference, retry, model switch, paid execution, or production mutation occurred during download.

## Qwen3 8B context4096 load-only

Authorization:

`G18-PHASEC-QWEN3-8B-CONTEXT4096-LOAD-SMOKE-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-qwen3-8b-context4096-load-smoke.ts`

Hardware boundary:
- system RAM: ~7.84 GiB;
- GPU VRAM: 4096 MiB;
- model artifact: ~5.23 GB;
- risk: `HIGH_MEMORY_PRESSURE_EXPECTED`.

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

Do not authorize context8192 or inference before reviewing the measured 4096 result.

## Exact next action

```text
EXECUTE_QWEN3_8B_CONTEXT4096_LOAD_ONLY_MEMORY_PREFLIGHT
```
