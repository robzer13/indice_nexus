# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-044`

## Granite 4.1 3B — context4096 load-only result

Status: `PASS_LOAD_ONLY_MEASURED`

Measured:
- before free RAM: 1.69 GiB;
- loaded free RAM: 1.39 GiB;
- after free RAM: 1.95 GiB;
- loaded VRAM used/free: 2321 / 1642 MiB;
- processor split: 15%/85% CPU/GPU;
- Ollama reported loaded size: 2.7 GB;
- load-only confirmed;
- explicit unload complete;
- post-unload VRAM used/free: 0 / 3962 MiB.

Interpretation:

`PASS_WITH_COMFORTABLE_RELATIVE_HEADROOM`

The 4096 authorization is consumed.

## Current authorized step

`EXECUTE_GRANITE4_1_3B_CONTEXT8192_LOAD_ONLY_MEMORY_PREFLIGHT`

Authorization:

`G18-PHASEC-GRANITE4_1-3B-CONTEXT8192-LOAD-SMOKE-AUTH-001`

One exact load-only execution at 8192 context tokens is authorized.

No semantic inference, retry, model switch or context growth beyond 8192 is authorized.

Protocol:

`calibration/vnext/OROTITAN_GATE18_PHASE_C_GRANITE4_1_3B_CONTEXT8192_LOAD_ONLY_PROTOCOL_001.json`

## Exact next action

`EXECUTE_GRANITE4_1_3B_CONTEXT8192_LOAD_ONLY_MEMORY_PREFLIGHT`
