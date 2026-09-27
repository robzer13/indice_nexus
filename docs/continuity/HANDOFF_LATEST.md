# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-021`

## Standing authority

`OROTITAN-STANDING-TECHNICAL-AUTH-002` remains ACTIVE for zero-cost technical execution. Nonzero external monetary cost still requires explicit user authorization.

## Gemma 3 target-context hardware qualification

Exact model:
- `gemma3:4b-it-q4_K_M`
- digest `a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`
- `Q4_K_M`

Context 16384 load-only result:

`PASS_LOAD_ONLY_MEASURED`

Measured:
- VRAM used: 2449 MiB;
- VRAM headroom: 1514 MiB;
- free system RAM while loaded: 0.33 GiB;
- processor split: 57%/43% CPU/GPU;
- load-only guard confirmed;
- explicit unload confirmed;
- full GPU-memory release after unload: yes;
- prompt provided: no;
- semantic inference: no.

Hardware disposition:

`PASS_WITH_HIGH_RAM_PRESSURE`

Context growth beyond 16384 is not recommended or authorized.

## First Gemma 3 C4 discriminator

Selected cell:

`Constellation Software / SERIAL_ACQUIRER`

Reason:
Use the identical Constellation packet already used for prior candidates to maximize diagnostic comparability on exact evidence grounding, conflict handling, weak-link usefulness, unresolved-point usefulness, and priority selection.

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
- generation prompt contract unchanged;
- generation schema contract unchanged;
- validation contract `GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1`.

Authorization:

`G18-PHASEC-C4-CONSTELLATION-GEMMA3-4B-V1_1-CONTEXT16384-OUTPUT1024-TIMEOUT600-LOOPBACK-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-c4-constellation-gemma3-4b-v1-1-context16384-output1024-timeout600-loopback-guarded.ts`

Execution boundaries:
- exactly one local inference;
- exact pinned model digest required;
- exact pinned private packet required;
- no automatic retry;
- no prompt/schema/packet/context/output/temperature/timeout change;
- generated content stays under `calibration/vnext/private-runs/`;
- human adjudication required if engineering PASS;
- no model-ranking authority;
- no routing authority;
- no production mutation;
- Gemma use remains ASSIST-only.

## Exact next action

```text
EXECUTE_FIRST_BOUNDED_GEMMA3_CONSTELLATION_C4_INFERENCE_V1_1
```

If the run is an engineering PASS, perform human-quality adjudication before any additional Gemma C4 cell.
