# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-046`

## Granite 4.1 3B — hardware qualification complete

Context16384 load-only status: `PASS_LOAD_ONLY_MEASURED`

Measured loaded state:
- free RAM: 0.84 GiB;
- VRAM used/free: 2305 / 1658 MiB;
- processor split: 38%/62% CPU/GPU;
- explicit unload complete;
- no semantic inference in the load-only step.

Hardware qualification:

`PASS_WITH_USEFUL_HEADROOM`

The context16384 load-only authorization is consumed.

## Current authorized step

`EXECUTE_FIRST_BOUNDED_GRANITE4_1_CONSTELLATION_C4_INFERENCE_V1_1`

Authorization:

`G18-PHASEC-C4-CONSTELLATION-GRANITE4_1-3B-V1_1-CONTEXT16384-OUTPUT1024-TIMEOUT600-LOOPBACK-AUTH-001`

Frozen execution contract:
- company: Constellation Software;
- packet SHA256: `9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8`;
- prompt SHA256: `0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8`;
- context: 16384;
- max output: 1024;
- temperature: 0;
- timeout: 600000 ms;
- minimum free RAM before inference: 1.0 GiB;
- local Ollama loopback only;
- one run only.

No automatic retry, prompt/packet/schema/context/output/temperature/timeout change, model switch, production mutation or public generated-content publication is authorized.

Human adjudication remains mandatory if engineering validation passes.

Runner:

`scripts/vnext-gate18-phase-c-c4-constellation-granite4-1-3b-v1-1-context16384-output1024-timeout600-loopback-guarded.ts`

## Exact next action

`EXECUTE_FIRST_BOUNDED_GRANITE4_1_CONSTELLATION_C4_INFERENCE_V1_1`
