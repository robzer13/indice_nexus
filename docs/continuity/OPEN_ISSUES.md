# OroTitan VNExT — Open Issues

## OI-001 — Completed/stopped candidate calibration evidence
Status: RETAINED_IMMUTABLE

## OI-002 — Granite 4.1 first Constellation C4
Status: COMPLETE_ENGINEERING_PASS_HUMAN_QUALITY_CRITICAL_FAILURE

Expansion: STOPPED
Retry: NOT_AUTHORIZED

## OI-003 — Llama 3.2 staged hardware qualification
Status: COMPLETE_PASS_WITH_HIGH_RAM_PRESSURE

Context growth stops at 16384.

## OI-004 — Llama 3.2 first bounded Constellation C4
Status: AUTHORIZED_UNCONSUMED_PRECONDITION_BLOCK
Priority: P0

Observed baseline free RAM: 0.75 GiB
Required baseline free RAM: >= 1.0 GiB

Provider generation reached: FALSE
Semantic inference executed: FALSE
Authorization consumed: FALSE
Authorized runs remaining: 1
Automatic retry: NOT_AUTHORIZED
Parameter change: NOT_AUTHORIZED

Required action:
restore baseline free RAM to at least 1.0 GiB, then manually execute the same single authorized runner once.

## OI-005 — Model winner and routing
Status: OPEN_GUARDED
