# OroTitan VNExT — Open Issues

## OI-001 — Phi-4, Qwen3.5 and Gemma 3 calibration paths
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

## OI-002 — Granite 4 3B pinned download
Status: COMPLETE_PASS_PINNED

Exact digest:
`89962fcc75239ac434cdebceb6b7e0669397f92eaef9c487774b718bc36a3e5f`

## OI-003 — Granite 4 3B context4096 load-only
Status: READY_LOCAL_EXECUTION
Priority: P0

Exactly one guarded load-only run is authorized. No prompt or inference.

## OI-004 — Granite 4 higher-context hardware fit
Status: BLOCKED_BY_OI-003

Decide context8192 only after the 4096 measurement.

## OI-005 — Granite 4 first C4 discriminator
Status: BLOCKED_BY_HARDWARE_QUALIFICATION

No inference authorized yet.

## OI-006 — Qwen3 8B post-Granite qualification
Status: QUEUED_BY_EXPLICIT_USER_DIRECTION

Qwen3 8B must be tested immediately after the Granite sequence, with staged hardware preflight before any inference.

## OI-007 — Ministral 3 / Llama 3.2 alternatives
Status: DEFERRED_UNTIL_AFTER_QWEN3_8B

## OI-008 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
