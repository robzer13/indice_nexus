# OroTitan VNExT — Open Issues

## OI-001 — v1.1 validation architecture

Status: COMPLETE

## OI-002 — Phi-4 C4 qualification

Status: COMPLETE_STOPPED_AFTER_HUMAN_CRITICAL_FAILURE

## OI-003 — Qwen3.5 target-context hardware qualification

Status: COMPLETE_FOR_TARGET_CONTEXT

Load-only PASS at 4096, 8192, and 16384.

## OI-004 — First Qwen3.5 Constellation attempt

Status: FAIL_PRESERVED_RUNTIME_ADAPTER_CONFOUNDED

The historical first run remains FAIL and is not reclassified.

## OI-005 — Clean same-cell Qwen3.5 retry

Status: COMPLETE_ENGINEERING_PASS_HUMAN_CRITICAL_FAILURE

The `think:false` retry passed engineering validation but failed human-quality adjudication on exact evidence grounding.

## OI-006 — Qwen3.5 human-quality adjudication

Status: COMPLETE_CRITICAL_FAILURE

The material defect is an E-045 replacement-RFP state overstated as completed replacement.

## OI-007 — Qwen3.5 expansion/stop disposition

Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

No broader Qwen3.5 C4 expansion. No global family failure inferred.

## OI-008 — Gemma 3 4B terms and static review

Status: COMPLETE_TERMS_ACCEPTED

Exact public tag verified as `gemma3:4b-it-q4_K_M`; static hardware fit remains plausible but unproven.

## OI-009 — Gemma 3 4B pinned download

Status: READY_LOCAL_EXECUTION
Priority: P0

Run exactly one pinned download and identity verification. No load or inference is authorized by the download runner.

## OI-010 — Gemma 3 4B context4096 load-only memory preflight

Status: BLOCKED_BY_OI-009

Prepare only after the full local model digest is captured from a successful download.

## OI-011 — Model winner and routing

Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
