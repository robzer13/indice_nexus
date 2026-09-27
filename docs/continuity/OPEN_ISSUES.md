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

Target context 16384 loads and unloads cleanly. No further context growth is authorized.

## OI-005 — Gemma 3 first Constellation C4 run
Status: COMPLETE_FAIL_DETERMINISTIC_SEMANTIC_CONTRACT

Observed:
- runtime error: none;
- schema valid: true;
- semantic valid: false;
- semantic error: `VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_WITHOUT_IDS`;
- v1.1 normalized path count: 13;
- substantive status: FAIL;
- output token margin: 341.

No retry is authorized.

## OI-006 — Gemma 3 Constellation deterministic semantic forensic
Status: READY_LOCAL_EXECUTION
Priority: P0

Run the prepared read-only forensic against the private artifact. No inference or source mutation.

## OI-007 — Gemma 3 post-Constellation disposition
Status: BLOCKED_BY_OI-006

Decide whether the first deterministic defect is isolated or whether additional semantic defects remain. Human-quality adjudication is not reached while engineering semantics fail.

## OI-008 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
