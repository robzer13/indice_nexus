# OroTitan VNExT — Open Issues

## OI-001 — v1.1 validation architecture
Status: COMPLETE

## OI-002 — Phi-4 C4 qualification
Status: COMPLETE_STOPPED_AFTER_HUMAN_CRITICAL_FAILURE

## OI-003 — Qwen3.5 qualification
Status: COMPLETE_STOPPED_RETAIN_CALIBRATION_EVIDENCE

No global Qwen3.5-family failure inferred.

## OI-004 — Gemma 3 terms/static review
Status: COMPLETE_TERMS_ACCEPTED

## OI-005 — Gemma 3 pinned download
Status: COMPLETE_PASS_PINNED

Exact digest:
`a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`

## OI-006 — Gemma 3 context4096 load-only
Status: COMPLETE_PASS_LOAD_ONLY_MEASURED

## OI-007 — Gemma 3 context8192 load-only
Status: COMPLETE_PASS_LOAD_ONLY_MEASURED

Measured:
- VRAM used 2423 MiB;
- VRAM headroom 1540 MiB;
- loaded free RAM 0.77 GiB;
- processor split 56%/44% CPU/GPU;
- full unload release confirmed;
- no semantic inference.

## OI-008 — Gemma 3 context16384 load-only
Status: READY_LOCAL_EXECUTION
Priority: P0

Exactly one guarded load-only run is authorized.

## OI-009 — Gemma 3 first C4 discriminator
Status: BLOCKED_BY_OI-008

No inference authorized yet.

## OI-010 — Model winner and routing
Status: OPEN_GUARDED

No winner selected. Routing remains unfrozen.
