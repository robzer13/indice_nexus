# OroTitan VNExT — Open Issues

## OI-001 — v1.1 validation architecture
Status: COMPLETE

## OI-002 — Phi-4 C4 qualification
Status: COMPLETE_STOPPED_AFTER_HUMAN_CRITICAL_FAILURE

## OI-003 — Qwen3.5 qualification
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

No global Qwen3.5-family failure inferred.

## OI-004 — Gemma 3 hardware qualification
Status: COMPLETE_PASS_WITH_HIGH_RAM_PRESSURE

## OI-005 — Gemma 3 first Constellation C4 run
Status: COMPLETE_FAIL_DETERMINISTIC_SEMANTIC_CONTRACT

First semantic error:
`VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_WITHOUT_IDS`

## OI-006 — Gemma 3 first deterministic forensic
Status: COMPLETE_ADDITIONAL_SEMANTIC_DEFECT_FOUND

Observed:
- all 3 priority findings had empty counterevidence IDs with non-null counterevidence links;
- after narrow in-memory normalization, downstream error:
  `VNEXT_GATE18_V10_UNKNOWN_CONFLICT_REF`.

## OI-007 — Gemma 3 unknown-conflict-reference forensic
Status: READY_LOCAL_EXECUTION
Priority: P0

Audit conflict references against the canonical packet, remove only unknown refs on an in-memory diagnostic copy, and rerun the frozen validator.

## OI-008 — Gemma 3 post-Constellation disposition
Status: BLOCKED_BY_OI-007

No retry or additional Gemma inference is authorized.

## OI-009 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
