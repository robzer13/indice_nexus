# OroTitan VNExT — Open Issues

## OI-001 — v1.1 validation architecture
Status: COMPLETE

## OI-002 — Phi-4 C4 qualification
Status: COMPLETE_STOPPED_AFTER_HUMAN_CRITICAL_FAILURE

## OI-003 — Qwen3.5 qualification
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

No global Qwen3.5-family failure inferred.

## OI-004 — Gemma 3 qualification
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

Observed deterministic defect classes on the first Constellation C4 run:
- non-null counterevidence links with empty counterevidence IDs on all three priority findings;
- invented unknown conflict ID `C-006` reused across one material conflict and two unresolved points.

After cumulative diagnostic-only normalization, the frozen validator passes. The historical run remains FAIL. Engineering PASS and human adjudication were not reached.

No global Gemma-family failure inferred.

## OI-005 — Local candidate registry refresh
Status: COMPLETE

Selected:
`GRANITE4_3B_OLLAMA_Q4_K_M`

Alternates retained:
- Ministral 3 3B Instruct Q4_K_M;
- Llama 3.2 3B;
- Qwen3 8B remains on poor-hardware-fit hold.

## OI-006 — Granite 4 3B pinned download
Status: READY_LOCAL_EXECUTION
Priority: P0

Exactly one zero-cost download of `granite4:3b` is authorized, followed by identity verification only.

## OI-007 — Granite 4 3B memory qualification
Status: BLOCKED_BY_OI-006

If download identity passes, prepare a separate context4096 load-only memory preflight.

## OI-008 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
