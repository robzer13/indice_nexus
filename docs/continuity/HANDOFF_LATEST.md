# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-033`

## Qwen3 8B context16384 result

Status:

`PASS_LOAD_ONLY_MEASURED`

Measured:
- free RAM before: 2.16 GiB;
- free RAM loaded: 0.14 GiB;
- free RAM after unload: 1.31 GiB;
- VRAM used loaded: 2345 MiB;
- VRAM free loaded: 1618 MiB;
- processor split: `70%/30% CPU/GPU`;
- Ollama resident size: 7.8 GB;
- explicit unload: complete;
- semantic inference: none.

Interpretation:

`PASS_WITH_EXTREME_RAM_PRESSURE`

This proves loadability at the exact comparable context but does not prove production fit or inference stability. No further context growth is allowed.

## One bounded Qwen3 8B C4 experiment

Selected cell:

`Constellation Software / SERIAL_ACQUIRER`

Pinned invariants:
- packet SHA256 `9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8`;
- prompt SHA256 `0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8`;
- prompt bytes 9041;
- context 16384;
- max output 1024;
- temperature 0;
- timeout 600000 ms;
- `think:false`;
- validation contract `GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1`.

Authorization:

`G18-PHASEC-C4-CONSTELLATION-QWEN3-8B-V1_1-CONTEXT16384-OUTPUT1024-TIMEOUT600-THINKFALSE-LOOPBACK-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-c4-constellation-qwen3-8b-v1-1-context16384-output1024-timeout600-thinkfalse-loopback-guarded.ts`

Additional memory guard:
- baseline free RAM must be at least 1.0 GiB before inference begins;
- otherwise the runner blocks before inference and does not consume a model result.

Boundaries:
- exactly one local inference;
- no automatic retry;
- no prompt/schema/packet/context/output/temperature/timeout/thinking-mode change;
- generated content remains private under `calibration/vnext/private-runs/`;
- human adjudication required if engineering PASS;
- no production-fit conclusion from hardware loadability;
- no model-ranking authority;
- no routing authority;
- no production mutation.

## Exact next action

```text
EXECUTE_ONE_BOUNDED_QWEN3_8B_CONSTELLATION_C4_THINK_FALSE_EXPERIMENT
```
