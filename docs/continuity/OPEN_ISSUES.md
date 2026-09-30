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

## OI-004 — Llama 3.2 context4096 load-only preflight
Status: COMPLETE_PASS_WITH_CRITICAL_RAM_PRESSURE

Measured loaded state:
- free RAM: 0.34 GiB;
- VRAM used: 2683 MiB;
- VRAM free: 1280 MiB;
- processor split: 20% CPU / 80% GPU.

The load-only authorization is consumed. No inference occurred.

## OI-005 — Model winner and routing
Status: OPEN_GUARDED

## OI-006 — Llama 3.2 context8192 load-only diagnostic
Status: AUTHORIZED_UNCONSUMED_CORRECTED_RUNNER_READY
Priority: P0

The first command was blocked before any Ollama call because the runner referenced the already-consumed context4096 authorization artifact.

Authorization consumption: FALSE
Authorized runs remaining: 1
Inference: NOT_AUTHORIZED
Automatic retry: NOT_AUTHORIZED
Context16384: NOT_PREAUTHORIZED

Required action:
execute the same context8192 load-only preflight once using the corrected runner, then submit the complete JSON result for adjudication.
