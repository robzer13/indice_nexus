# OroTitan VNExT — Open Issues

## OI-001 — Phi-4, Qwen3.5 and Gemma 3 calibration paths
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

## OI-002 — Granite 4 staged hardware qualification
Status: COMPLETE_PASS_WITH_HIGH_RAM_PRESSURE

Context4096, 8192 and 16384 load-only runs all passed. At 16384:
- loaded free RAM: 0.41 GiB;
- loaded free VRAM: 1658 MiB;
- processor split: 38%/62% CPU/GPU;
- explicit unload: complete.

No context growth beyond 16384 is authorized.

## OI-003 — Granite 4 first Constellation C4 inference
Status: READY_LOCAL_EXECUTION
Priority: P0

Exactly one same-packet local inference is authorized at context16384.

## OI-004 — Granite 4 human-quality adjudication
Status: BLOCKED_BY_OI-003

Required only if engineering semantics pass.

## OI-005 — Qwen3 8B post-Granite qualification
Status: QUEUED_BY_EXPLICIT_USER_DIRECTION

Qwen3 8B is the next candidate immediately after the Granite result, using staged hardware preflight.

## OI-006 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
