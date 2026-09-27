# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-045`

## Granite 4.1 3B — context8192 load-only result

Status: `PASS_LOAD_ONLY_MEASURED`

Measured:
- before free RAM: 2.09 GiB;
- loaded free RAM: 0.91 GiB;
- after free RAM: 1.68 GiB;
- loaded VRAM used/free: 2355 / 1608 MiB;
- processor split: 22%/78% CPU/GPU;
- Ollama reported loaded size: 3.1 GB;
- load-only confirmed;
- explicit unload complete;
- post-unload VRAM used/free: 0 / 3962 MiB.

Interpretation:

`PASS_WITH_USEFUL_HEADROOM`

The 8192 authorization is consumed.

The 0.03 GiB loaded-RAM difference versus Granite 4 3B at 8192 is treated as host-state-sensitive rather than structural because VRAM use and processor split are identical.

## Current authorized step

`EXECUTE_GRANITE4_1_3B_CONTEXT16384_LOAD_ONLY_MEMORY_PREFLIGHT`

Authorization:

`G18-PHASEC-GRANITE4_1-3B-CONTEXT16384-LOAD-SMOKE-AUTH-001`

One exact load-only execution at 16384 context tokens is authorized.

No semantic inference, retry, model switch or context growth beyond 16384 is authorized.

Protocol:

`calibration/vnext/OROTITAN_GATE18_PHASE_C_GRANITE4_1_3B_CONTEXT16384_LOAD_ONLY_PROTOCOL_001.json`

## Exact next action

`EXECUTE_GRANITE4_1_3B_CONTEXT16384_LOAD_ONLY_MEMORY_PREFLIGHT`
