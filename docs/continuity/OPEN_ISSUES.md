# OroTitan VNExT — Open Issues

## OI-001 — Completed/stopped candidate calibration evidence
Status: RETAINED_IMMUTABLE

## OI-002 — Llama 3.2 staged hardware qualification
Status: COMPLETE_PASS_WITH_HIGH_RAM_PRESSURE

## OI-003 — Llama 3.2 first Constellation C4
Status: FAIL_RAW_SCHEMA_FORENSICS_IN_PROGRESS
Priority: P0

Historical inference:
- consumed exactly one authorized inference;
- runtime completed;
- 895 / 1024 generated tokens;
- raw JSON syntactically valid;
- raw V1.1 schema failed;
- semantics and human adjudication not reached.

First read-only forensic:
- COMPLETE;
- exactly one schema issue;
- path `priority_findings.1.counterevidence_ids.0`;
- code `invalid_format`;
- format `regex`.

## OI-004 — Llama 3.2 counterevidence-ID forensic
Status: AUTHORIZED_UNCONSUMED
Priority: P0

Exactly one local read-only forensic is authorized.

It may inspect only the malformed ID structure and canonical packet IDs and may perform one in-memory diagnostic replacement if the decomposition is unambiguous.

It may not:
- print the malformed raw value;
- print narrative content;
- mutate the source artifact;
- call Ollama;
- use network access;
- execute model inference;
- authorize a retry.

Additional Llama inference authorized: FALSE

## OI-005 — Model winner and routing
Status: OPEN_GUARDED
