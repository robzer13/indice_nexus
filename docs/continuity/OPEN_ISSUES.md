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

The historical run remains FAIL. Its persisted final response is exactly empty while 842 tokens were evaluated. The original runner omitted explicit thinking-mode control and did not persist provider thinking output.

## OI-005 — Clean same-cell Qwen3.5 retry

Status: COMPLETE_ENGINEERING_PASS_HUMAN_ADJUDICATION_PENDING

The one authorized same-cell retry with `think:false` was consumed. It passed schema, raw-presentation, and deterministic semantic validation with zero safe-normalization paths. No automatic second retry is authorized.

## OI-006 — Qwen3.5 human-quality adjudication

Status: READY_PRIVATE_REVIEW
Priority: P0

Review the private raw output as emitted against the pinned Constellation evidence packet. Do not infer a clean qualification result from the engineering PASS alone.

## OI-007 — Qwen3.5 expansion/stop disposition

Status: BLOCKED_BY_OI-006

Decide expansion or stop only after human-quality adjudication of the Constellation discriminator.

## OI-008 — Model winner and routing

Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
