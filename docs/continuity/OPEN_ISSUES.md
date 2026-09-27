# OroTitan VNExT — Open Issues

## OI-001 — Prior local candidate calibration paths
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

## OI-002 — Ministral 3 3B pinned download
Status: COMPLETE_PASS_PINNED

## OI-003 — Ministral 3 3B context4096
Status: COMPLETE_PASS_WITH_CRITICAL_RAM_PRESSURE

## OI-004 — Ministral 3 3B context8192
Status: COMPLETE_PASS_WITH_CRITICAL_RAM_PRESSURE

Measured:
- loaded free RAM: 0.38 GiB;
- loaded free VRAM: 1556 MiB;
- processor split: 54%/46% CPU/GPU;
- unload complete.

This is not inference qualification.

## OI-005 — Ministral 3 3B context16384 final diagnostic
Status: READY_LOCAL_EXECUTION
Priority: P0

Exactly one guarded context16384 load-only run is authorized. No prompt or inference.

## OI-006 — Ministral 3 3B semantic inference
Status: BLOCKED_BY_OI-005

No inference is authorized before explicit review of context16384.

## OI-007 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
