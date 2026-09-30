# OroTitan VNExT — Open Issues

## OI-001 — Completed/stopped candidate calibration evidence
Status: RETAINED_IMMUTABLE

## OI-002 — Llama 3.2 staged hardware qualification
Status: COMPLETE_PASS_WITH_HIGH_RAM_PRESSURE

## OI-003 — Llama 3.2 first bounded Constellation C4
Status: AUTHORIZED_UNCONSUMED_PRECONDITION_BLOCK
Priority: P0

Precondition blocks recorded: 2

Latest external pre-launch free RAM: 1.09 GiB
Latest runner baseline free RAM: 0.70 GiB
Runner protocol requirement: >= 1.00 GiB
Operational external pre-launch target: >= 1.50 GiB

Provider generation reached: FALSE
Semantic inference executed: FALSE
Authorization consumed: FALSE
Authorized runs remaining: 1
Automatic retry: NOT_AUTHORIZED
Parameter change: NOT_AUTHORIZED

Required action:
restore external free RAM to at least 1.5 GiB before launch, then manually execute the same single authorized runner once.

## OI-004 — Model winner and routing
Status: OPEN_GUARDED
