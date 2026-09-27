# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-028`

## Granite 4 target-context hardware qualification

Exact model:
- `granite4:3b`
- digest `89962fcc75239ac434cdebceb6b7e0669397f92eaef9c487774b718bc36a3e5f`
- `Q4_K_M`

Context16384 load-only result:

`PASS_LOAD_ONLY_MEASURED`

Measured:
- VRAM used: 2305 MiB;
- VRAM headroom: 1658 MiB;
- free system RAM while loaded: 0.41 GiB;
- processor split: 38%/62% CPU/GPU;
- load-only guard confirmed;
- explicit unload confirmed;
- full GPU-memory release after unload: yes;
- prompt provided: no;
- semantic inference: no.

Hardware disposition:

`PASS_WITH_HIGH_RAM_PRESSURE`

No context growth beyond 16384 is authorized.

## First Granite 4 C4 discriminator

Selected cell:

`Constellation Software / SERIAL_ACQUIRER`

Pinned invariants:
- packet SHA256 `9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8`;
- prompt SHA256 `0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8`;
- prompt bytes 9041;
- evidence count 11;
- conflict count 1;
- context 16384;
- max output 1024;
- temperature 0;
- timeout 600000 ms;
- generation prompt/schema unchanged;
- validation contract `GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1`.

Authorization:

`G18-PHASEC-C4-CONSTELLATION-GRANITE4-3B-V1_1-CONTEXT16384-OUTPUT1024-TIMEOUT600-LOOPBACK-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-c4-constellation-granite4-3b-v1-1-context16384-output1024-timeout600-loopback-guarded.ts`

Execution boundaries:
- exactly one local inference;
- exact pinned model digest required;
- exact pinned private packet required;
- no automatic retry;
- no prompt/schema/packet/context/output/temperature/timeout change;
- generated content remains under `calibration/vnext/private-runs/`;
- human adjudication required if engineering PASS;
- no model-ranking authority;
- no routing authority;
- no production mutation.

## User-directed next candidate

Qwen3 8B remains immediately after the Granite result. No Ministral/Llama detour is permitted before that user-directed test.

## Exact next action

```text
EXECUTE_FIRST_BOUNDED_GRANITE4_CONSTELLATION_C4_INFERENCE_V1_1
```
