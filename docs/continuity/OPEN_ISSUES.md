# OroTitan VNExT — Open Issues

## OI-001 — Prior local candidate calibration paths
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

## OI-002 — Ministral 3 3B staged hardware qualification
Status: COMPLETE_PASS_WITH_USEFUL_HEADROOM_AT_CONTEXT16384

Measured context16384:
- loaded free RAM: 1.21 GiB;
- loaded free VRAM: 1564 MiB;
- processor split: 63%/37% CPU/GPU;
- explicit unload: complete.

This proves loadability, not production fit.

## OI-003 — Ministral 3 3B first bounded Constellation C4 inference
Status: READY_LOCAL_EXECUTION
Priority: P0

Exactly one same-packet context16384 inference is authorized.

Required baseline guard:
- free RAM >= 1.0 GiB;
- no other loaded Ollama model.

No automatic retry or parameter change is authorized.

## OI-004 — Ministral 3 3B human adjudication
Status: BLOCKED_BY_OI-003

Required only if the engineering/validation run completes sufficiently for human-quality review.

## OI-005 — Ministral 3 3B C4 expansion
Status: NOT_AUTHORIZED

No second company/cell or retry is pre-authorized.

## OI-006 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
