# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-039`

## Ministral 3 3B context16384 result

Status:

`PASS_LOAD_ONLY_MEASURED`

Measured:
- free RAM before: 1.45 GiB;
- free RAM loaded: 1.21 GiB;
- free RAM after unload: 3.88 GiB;
- VRAM used loaded: 2399 MiB;
- VRAM free loaded: 1564 MiB;
- processor split: `63%/37% CPU/GPU`;
- Ollama resident size: 4.4 GB;
- explicit unload: complete;
- semantic inference: none.

Interpretation:

`PASS_WITH_USEFUL_HEADROOM`

The absolute free-RAM comparison across runs is host-state-sensitive, so it must not be treated as an intrinsic memory-footprint estimate. The relevant conclusion is narrower: context16384 loaded and unloaded cleanly, retained 1.21 GiB free RAM in the measured state, and supports one bounded C4 capability test.

No context growth beyond 16384 is authorized.

## First bounded Ministral C4 inference

Selected cell:
- company: Constellation Software;
- archetype: SERIAL_ACQUIRER;
- module: `MOAT_EVIDENCE_AUDIT_ASSISTED_V0_3`;
- exact same pinned packet and prompt as prior candidates;
- context: 16384;
- max output: 1024;
- temperature: 0;
- timeout: 600000 ms;
- keep-alive: 0s.

Authorization:

`G18-PHASEC-C4-CONSTELLATION-MINISTRAL3-3B-V1_1-CONTEXT16384-OUTPUT1024-TIMEOUT600-LOOPBACK-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-c4-constellation-ministral3-3b-v1-1-context16384-output1024-timeout600-loopback-guarded.ts`

Pre-inference guards:
- exact full model digest;
- no other Ollama model loaded;
- free system RAM must be at least 1.0 GiB;
- exact packet and prompt hashes;
- loopback transport only;
- Windows sleep guard;
- one authorized inference only;
- no automatic retry;
- no prompt/schema/packet/context/output/temperature/timeout/model change;
- generated content remains private;
- no production or routing authority.

## Exact next action

```text
EXECUTE_FIRST_BOUNDED_MINISTRAL3_CONSTELLATION_C4_INFERENCE_V1_1
```
