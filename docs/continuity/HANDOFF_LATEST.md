# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260928-049`

## Llama 3.2 3B — pinned download complete

Status: `PASS_PINNED_DOWNLOAD_ONLY`

Observed local identity:
- model: `llama3.2:3b-instruct-q4_K_M`;
- digest: `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`;
- size: 2019393189 bytes;
- format: gguf;
- family: llama;
- Ollama-reported parameter size: 3.2B;
- quantization: Q4_K_M.

Verification:
- exact tag present;
- expected digest prefix matched;
- API show reachable;
- expected quantization matched.

No model load or inference occurred in the download step.

The download authorization is consumed.

## Current authorized step

`EXECUTE_LLAMA3_2_3B_CONTEXT4096_LOAD_ONLY_MEMORY_PREFLIGHT`

Authorization:

`G18-PHASEC-LLAMA3_2-3B-LOAD-SMOKE-AUTH-001`

Protocol:

`calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT4096_LOAD_ONLY_PROTOCOL_001.json`

Exactly one local load-only execution at 4096 context tokens is authorized.

No prompt, semantic inference, retry, context change, model switch, production mutation or publication authority is granted.

## Exact next action

`EXECUTE_LLAMA3_2_3B_CONTEXT4096_LOAD_ONLY_MEMORY_PREFLIGHT`
