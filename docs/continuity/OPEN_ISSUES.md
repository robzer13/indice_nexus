# OroTitan VNExT — Open Issues

## OI-001 — Completed/stopped candidate calibration evidence
Status: RETAINED_IMMUTABLE

## OI-002 — Llama 3.2 staged hardware qualification
Status: COMPLETE_PASS_WITH_HIGH_RAM_PRESSURE

## OI-003 — Llama 3.2 first Constellation C4
Status: FAIL_RAW_SCHEMA_FORENSICS_IN_PROGRESS
Priority: P0

Historical run:
- inference executed once and authorization consumed;
- raw JSON syntactically valid;
- exactly one raw-schema issue;
- path `priority_findings.1.counterevidence_ids.0`.

Second forensic:
- COMPLETE;
- malformed value length: 5;
- exact `E-NNN` tokens: 0;
- deterministic token decomposition: not admissible.

## OI-004 — Llama 3.2 prefix/separator forensic
Status: AUTHORIZED_UNCONSUMED
Priority: P0

Allowed:
- inspect suffix positions 2-4;
- test whether digits already identify one canonical packet ID;
- normalize only prefix/separator on an in-memory copy;
- rerun V1.1 on that copy.

Forbidden:
- digit substitution;
- source mutation;
- raw malformed value publication;
- inference;
- retry authorization.

Additional Llama inference authorized: FALSE

## OI-005 — Model winner and routing
Status: OPEN_GUARDED
