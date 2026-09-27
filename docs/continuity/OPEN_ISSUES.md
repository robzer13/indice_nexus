# OroTitan VNExT — Open Issues

## OI-001 — Prior local candidate calibration paths
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

## OI-002 — Ministral 3 3B pinned download
Status: COMPLETE_PASS_PINNED

## OI-003 — Ministral 3 3B context4096
Status: COMPLETE_PASS_WITH_CRITICAL_RAM_PRESSURE

Measured:
- loaded free RAM: 0.35 GiB;
- loaded free VRAM: 1566 MiB;
- processor split: 48%/52% CPU/GPU;
- explicit unload: complete.

This is not inference qualification.

## OI-004 — Ministral 3 3B context8192 diagnostic load-only
Status: READY_LOCAL_EXECUTION
Priority: P0

Exactly one guarded context8192 load-only run is authorized. No prompt or inference.

## OI-005 — Ministral 3 3B context16384
Status: NOT_AUTHORIZED

Any further context growth requires review of the 8192 measurement.

## OI-006 — Ministral 3 3B semantic inference
Status: NOT_AUTHORIZED

No inference until hardware fit is explicitly re-evaluated.

## OI-007 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
