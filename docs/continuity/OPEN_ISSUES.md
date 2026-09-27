# OroTitan VNExT — Open Issues

## OI-001 — v1.1 validation architecture

Status: COMPLETE

## OI-002 — Phi-4 C4 qualification

Status: COMPLETE_STOPPED_AFTER_HUMAN_CRITICAL_FAILURE

## OI-003 — Qwen3.5 target-context hardware qualification

Status: COMPLETE_FOR_TARGET_CONTEXT

## OI-004 — Qwen3.5 qualification disposition

Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

No global Qwen3.5-family failure inferred.

## OI-005 — Gemma 3 4B terms/static review

Status: COMPLETE_TERMS_ACCEPTED

## OI-006 — Gemma 3 4B pinned download

Status: COMPLETE_PASS_PINNED

Exact full digest:

`a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`

Observed local identity:
- 3,338,801,804 bytes;
- `gguf`;
- family `gemma3`;
- parameter size `4.3B`;
- `Q4_K_M`.

## OI-007 — Gemma 3 context4096 load-only memory preflight

Status: READY_LOCAL_EXECUTION
Priority: P0

Exactly one load-only run is authorized. No prompt or semantic inference.

## OI-008 — Gemma 3 higher-context hardware fit

Status: BLOCKED_BY_OI-007

Decide whether context8192 load-only is justified only after the measured context4096 result.

## OI-009 — Gemma 3 first C4 discriminator

Status: BLOCKED_BY_HARDWARE_QUALIFICATION

No inference authorized yet.

## OI-010 — Model winner and routing

Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
