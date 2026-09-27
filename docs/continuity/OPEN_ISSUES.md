# OroTitan VNExT — Open Issues

## OI-001 — Prior local candidate calibration paths
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

## OI-002 — Qwen3 8B pinned download
Status: COMPLETE_PASS_PINNED

## OI-003 — Qwen3 8B context4096
Status: COMPLETE_PASS_WITH_CRITICAL_RAM_PRESSURE

## OI-004 — Qwen3 8B context8192
Status: COMPLETE_PASS_WITH_CRITICAL_RAM_PRESSURE

Measured:
- loaded free RAM: 0.22 GiB;
- loaded free VRAM: 1640 MiB;
- processor split: 64%/36% CPU/GPU;
- unload complete.

## OI-005 — Qwen3 8B context16384 final diagnostic
Status: READY_LOCAL_EXECUTION
Priority: P0

Exactly one guarded load-only run is authorized.

## OI-006 — Qwen3 8B semantic inference
Status: BLOCKED_BY_OI-005

No inference is authorized before explicit review of context16384.

## OI-007 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
