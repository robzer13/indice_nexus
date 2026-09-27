# OroTitan VNExT — Open Issues

## OI-001 — Phi-4, Qwen3.5, Gemma 3 and Granite 4 calibration paths
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

## OI-002 — Qwen3 8B pinned download
Status: COMPLETE_PASS_PINNED

Exact identity:
- `qwen3:8b-q4_K_M`;
- digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`;
- 5,225,388,164 bytes;
- 8.2B;
- Q4_K_M.

## OI-003 — Qwen3 8B context4096 load-only
Status: READY_LOCAL_EXECUTION
Priority: P0

Exactly one guarded context4096 load-only run is authorized. No prompt or inference.

## OI-004 — Qwen3 8B higher-context hardware fit
Status: BLOCKED_BY_OI-003

Do not authorize context8192 until the 4096 measurement is reviewed.

## OI-005 — Qwen3 8B first C4 discriminator
Status: BLOCKED_BY_HARDWARE_QUALIFICATION

## OI-006 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
