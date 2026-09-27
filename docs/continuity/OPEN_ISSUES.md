# OroTitan VNExT — Open Issues

## OI-001 — v1.1 validation architecture

Status: COMPLETE

## OI-002 — Phi-4 C4 qualification

Status: COMPLETE_STOPPED_AFTER_HUMAN_CRITICAL_FAILURE

Phi-4 remains calibration evidence.

## OI-003 — Qwen3.5 4B hardware qualification

Status: COMPLETE_FOR_TARGET_CONTEXT

Load-only PASS at context 4096, 8192, and 16384.

## OI-004 — First Qwen3.5 C4 Constellation run

Status: FAIL_RAW_JSON_INCOMPLETE_FORENSICS_REQUIRED  
Priority: P0

Observed:

- runtime completed;
- done reason `stop`;
- eval count 842 / 1024;
- runtime error none;
- JSON parse failed with `Unexpected end of JSON input`;
- deterministic semantic validation was not reached.

## OI-005 — Qwen3.5 raw JSON termination forensic

Status: READY_LOCAL_EXECUTION  
Priority: P0

Read the existing private artifact only and measure:

- required-section progress;
- terminal string/delimiter state;
- structural closure feasibility on an in-memory copy;
- whether the output ended before completing required schema sections.

No inference, Ollama call, network access, raw-content publication, or source mutation.

## OI-006 — Qwen3.5 retry/stop disposition

Status: BLOCKED_BY_OI-005

After forensic results, decide between:

- one bounded same-cell reliability retry if diagnostically informative; or
- stop Qwen3.5 as insufficiently reliable for structured-output C4.

No automatic retry is currently authorized by the cell artifact.

## OI-007 — Model winner and routing

Status: OPEN_GUARDED

No winner is selected and routing remains unfrozen.
