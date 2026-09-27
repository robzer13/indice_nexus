# OroTitan VNExT — Open Issues

## OI-001 — Prior local candidate calibration paths
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

## OI-002 — Qwen3 8B staged hardware qualification
Status: COMPLETE_LOADABLE_WITH_EXTREME_RAM_PRESSURE

4096, 8192 and 16384 load-only runs passed. At 16384:
- loaded free RAM: 0.14 GiB;
- loaded free VRAM: 1618 MiB;
- processor split: 70%/30% CPU/GPU;
- resident size: 7.8 GB;
- explicit unload: complete.

No further context growth is authorized.

## OI-003 — Qwen3 8B first Constellation C4 experiment
Status: READY_LOCAL_EXECUTION
Priority: P0

Exactly one same-packet local inference is authorized with `think:false` and a 1.0 GiB pre-inference baseline-free-RAM guard.

## OI-004 — Qwen3 8B human-quality adjudication
Status: BLOCKED_BY_OI-003

Required only if engineering semantics pass.

## OI-005 — Qwen3 8B production fit
Status: NOT_ESTABLISHED

Hardware loadability under extreme RAM pressure must not be interpreted as production suitability.

## OI-006 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
