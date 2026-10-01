# OroTitan VNExT — Open Issues

## OI-001 — Completed/stopped candidate calibration evidence
Status: RETAINED_IMMUTABLE

## OI-002 — Llama 3.2 staged hardware qualification
Status: COMPLETE_PASS_WITH_HIGH_RAM_PRESSURE

## OI-003 — Llama 3.2 first bounded Constellation C4
Status: AUTHORIZED_UNCONSUMED_TOOLING_GUARD_FIXED
Priority: P0

Historical pre-inference blocks:
1. baseline RAM 0.75 GiB < 1.0 GiB;
2. baseline RAM 0.70 GiB < 1.0 GiB;
3. Ollama residency CLI guard failed before model identity/RAM/generation.

Current runner residency guard:
`GET /api/ps` over loopback HTTP.

Inference authorization consumed: FALSE
Authorized runs remaining: 1
Automatic retry: NOT_AUTHORIZED
Parameter change: NOT_AUTHORIZED

Operational launch condition:
- prefer external free RAM >= 1.5 GiB;
- no resident Ollama model.

## OI-004 — Model winner and routing
Status: OPEN_GUARDED
