# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-043`

## Granite 4.1 3B — pinned download result

Status: `PASS_PINNED_DOWNLOAD_ONLY`

Exact local identity:
- model: `granite4.1:3b-q4_K_M`;
- digest: `6fd349357287c7ffc9e38189a93b48ea175d24fc566b38f09cfc564fb7f303eb`;
- size: 2099520281 bytes;
- format: GGUF;
- family: granite;
- Ollama-reported parameter size: 3.4B;
- quantization: Q4_K_M.

The download authorization is consumed. No inference occurred.

## Current authorized step

`EXECUTE_GRANITE4_1_3B_CONTEXT4096_LOAD_ONLY_MEMORY_PREFLIGHT`

Authorization: `G18-PHASEC-GRANITE4_1-3B-LOAD-SMOKE-AUTH-001`

Target:
- exact pinned digest above;
- context 4096;
- load only;
- resource snapshots before / loaded / after;
- explicit unload.

No semantic inference, retry, context growth or model switch is authorized.

Execution contract:

`calibration/vnext/OROTITAN_GATE18_PHASE_C_GRANITE4_1_3B_CONTEXT4096_LOAD_ONLY_PROTOCOL_001.json`

## Deferred candidate

`LLAMA3_2_3B_OLLAMA_Q4_K_M` remains deferred after Granite 4.1.

## Exact next action

`EXECUTE_GRANITE4_1_3B_CONTEXT4096_LOAD_ONLY_MEMORY_PREFLIGHT`
