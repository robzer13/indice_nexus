# OroTitan VNExT — Open Issues

## OI-001 — Phi-4, Qwen3.5 and Gemma 3 calibration paths
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

## OI-002 — Granite 4 3B pinned download
Status: COMPLETE_PASS_PINNED

## OI-003 — Granite 4 context4096 load-only
Status: COMPLETE_PASS_LOAD_ONLY_MEASURED

Measured headroom:
- loaded free RAM: 1.06 GiB;
- loaded free VRAM: 1650 MiB;
- processor split: 15%/85% CPU/GPU;
- explicit unload: complete.

## OI-004 — Granite 4 context8192 load-only
Status: READY_LOCAL_EXECUTION
Priority: P0

Exactly one guarded context8192 load-only run is authorized. No prompt or inference.

## OI-005 — Granite 4 context16384 decision
Status: BLOCKED_BY_OI-004

Do not authorize context16384 until the 8192 measurement is reviewed.

## OI-006 — Granite 4 first C4 discriminator
Status: BLOCKED_BY_HARDWARE_QUALIFICATION

No inference authorized yet.

## OI-007 — Qwen3 8B post-Granite qualification
Status: QUEUED_BY_EXPLICIT_USER_DIRECTION

Qwen3 8B remains next immediately after the Granite sequence, with staged hardware preflight before inference.

## OI-008 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
