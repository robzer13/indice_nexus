# OroTitan VNExT — Open Issues

## OI-001 — Phi-4, Qwen3.5, Gemma 3 and Granite 4 calibration paths
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

Granite 4 reached engineering PASS on Constellation but failed human-quality adjudication critically. No global family failure inferred.

## OI-002 — Qwen3 8B pinned download
Status: READY_LOCAL_EXECUTION
Priority: P0

Exactly one zero-cost download of `qwen3:8b-q4_K_M` is authorized, followed by identity verification only.

## OI-003 — Qwen3 8B memory qualification
Status: BLOCKED_BY_OI-002

If download identity passes, prepare a separate context4096 load-only memory preflight. No direct inference.

## OI-004 — Qwen3 8B first C4 discriminator
Status: BLOCKED_BY_HARDWARE_QUALIFICATION

## OI-005 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
