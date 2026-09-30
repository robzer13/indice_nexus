# OroTitan VNExT — Open Issues

## OI-001 — Completed/stopped candidate calibration evidence
Status: RETAINED_IMMUTABLE

## OI-002 — Granite 4.1 first Constellation C4
Status: COMPLETE_ENGINEERING_PASS_HUMAN_QUALITY_CRITICAL_FAILURE

Expansion: STOPPED
Retry: NOT_AUTHORIZED

## OI-003 — Llama 3.2 3B pinned download
Status: COMPLETE_PASS

Exact digest:
`a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`

## OI-004 — Llama 3.2 staged load qualification
Status: COMPLETE_PASS_WITH_HIGH_RAM_PRESSURE

4096:
- free RAM loaded: 0.34 GiB;
- VRAM free: 1280 MiB;
- processor: 20% CPU / 80% GPU.

8192:
- free RAM loaded: 0.47 GiB;
- VRAM free: 1212 MiB;
- processor: 32% CPU / 68% GPU.

16384:
- free RAM loaded: 0.52 GiB;
- VRAM free: 1217 MiB;
- processor: 46% CPU / 54% GPU.

All three load-only authorizations are consumed.
No semantic inference occurred during load qualification.
No context growth beyond 16384 is authorized.

## OI-005 — Model winner and routing
Status: OPEN_GUARDED

## OI-006 — Llama 3.2 first bounded Constellation C4
Status: AUTHORIZED_UNCONSUMED
Priority: P0

Exactly one local inference is authorized.

Required baseline free RAM: >= 1.0 GiB
Context: 16384
Max output: 1024
Temperature: 0
Timeout: 600000 ms
Automatic retry: NOT_AUTHORIZED
Parameter change: NOT_AUTHORIZED
Production/publication: NOT_AUTHORIZED

Required action:
execute the exact guarded runner once, preserve generated content only in the private run destination, and submit the non-private summary output for adjudication.
