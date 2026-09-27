# OroTitan VNExT — Open Issues

## OI-001 — v1.1 validation architecture
Status: COMPLETE

## OI-002 — Phi-4 C4 qualification
Status: COMPLETE_STOPPED_AFTER_HUMAN_CRITICAL_FAILURE

## OI-003 — Qwen3.5 qualification
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

No global Qwen3.5-family failure inferred.

## OI-004 — Gemma 3 terms/static review
Status: COMPLETE_TERMS_ACCEPTED

## OI-005 — Gemma 3 pinned download
Status: COMPLETE_PASS_PINNED

## OI-006 — Gemma 3 context4096 load-only
Status: COMPLETE_PASS_LOAD_ONLY_MEASURED

## OI-007 — Gemma 3 context8192 load-only
Status: COMPLETE_PASS_LOAD_ONLY_MEASURED

## OI-008 — Gemma 3 context16384 load-only
Status: COMPLETE_PASS_WITH_HIGH_RAM_PRESSURE

Measured:
- VRAM used 2449 MiB;
- VRAM headroom 1514 MiB;
- loaded free RAM 0.33 GiB;
- processor split 57%/43% CPU/GPU;
- full unload release confirmed;
- no semantic inference.

No further context growth is authorized.

## OI-009 — Gemma 3 first C4 discriminator
Status: READY_LOCAL_EXECUTION
Priority: P0

Selected:
`Constellation Software / SERIAL_ACQUIRER`

Exactly one same-packet local inference is authorized. Human adjudication is mandatory after an engineering PASS.

## OI-010 — Gemma 3 post-Constellation disposition
Status: BLOCKED_BY_OI-009

No additional Gemma C4 cell is authorized until the first result and required human adjudication are complete.

## OI-011 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
