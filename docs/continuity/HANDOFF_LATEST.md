# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-043`

## Granite 4.1 3B — pinned download result

Status:

`PASS_PINNED_DOWNLOAD_ONLY`

Exact local identity:
- model: `granite4.1:3b-q4_K_M`;
- digest: `6fd349357287c7ffc9e38189a93b48ea175d24fc566b38f09cfc564fb7f303eb`;
- size: 2099520281 bytes;
- format: GGUF;
- family: granite;
- Ollama-reported parameter size: 3.4B;
- quantization: Q4_K_M.

Verification:
- exact tag present;
- expected digest prefix matched;
- API show reachable;
- expected quantization matched.

Safety:
- one exact download executed;
- no other model downloaded;
- no load smoke executed in the download step;
- no prompt;
- no inference;
- no retry;
- no model switch;
- no paid execution;
- no production mutation;
- no publication authority.

## Current authorized step

`EXECUTE_GRANITE4_1_3B_CONTEXT4096_LOAD_ONLY_MEMORY_PREFLIGHT`

Authorization:

`G18-PHASEC-GRANITE4_1-3B-LOAD-SMOKE-AUTH-001`

Target:
- exact pinned digest above;
- context 4096;
- local load only;
- keep-alive 2m;
- explicit unload;
- no prompt field;
- no semantic inference.

Because repository executable-script publication for this load call was blocked by connector safety controls, the exact equivalent loopback procedure is frozen in:

`calibration/vnext/OROTITAN_GATE18_PHASE_C_GRANITE4_1_3B_CONTEXT4096_LOAD_ONLY_PROTOCOL_001.json`

No retry, context growth or inference is pre-authorized.

## Deferred candidate

`LLAMA3_2_3B_OLLAMA_Q4_K_M` remains deferred after Granite 4.1.

## Exact next action

```text
EXECUTE_GRANITE4_1_3B_CONTEXT4096_LOAD_ONLY_MEMORY_PREFLIGHT
```
