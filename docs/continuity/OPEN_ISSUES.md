# OroTitan VNExT — Open Issues

## OI-001 — Completed/stopped candidate calibration evidence
Status: RETAINED_IMMUTABLE

## OI-002 — Llama 3.2 staged hardware qualification
Status: COMPLETE_PASS_WITH_HIGH_RAM_PRESSURE

## OI-003 — Llama 3.2 first bounded Constellation C4
Status: FAIL_RAW_SCHEMA_FORENSICS_REQUIRED
Priority: P0

Runtime execution completed:
- wall clock: 299230 ms;
- prompt tokens: 3157;
- generated tokens: 895 / 1024;
- done reason: stop;
- runtime error: null.

Validation:
- raw schema: FAIL;
- error: `VNEXT_GATE18_V11_RAW_SCHEMA_INVALID`;
- semantics: NOT EVALUATED;
- human adjudication: NOT REACHED.

Inference authorization consumed: TRUE
Additional inference authorized: FALSE
Automatic retry authorized: FALSE

## OI-004 — Llama 3.2 raw-schema forensic
Status: AUTHORIZED_UNCONSUMED
Priority: P0

Exactly one read-only local forensic is authorized.

No Ollama call.
No inference.
No network.
No raw generated values printed.
No source mutation.
No retry authorization.

Required action:
execute the prepared forensic against the exact private artifact from the failed C4 run and return only the sanitized JSON summary.

## OI-005 — Model winner and routing
Status: OPEN_GUARDED
