# OroTitan VNExT — Open Issues

## OI-001 — Execute v1.1 shadow replay

Status: OPEN  
Priority: P0  
Inference required: NO

Run the prepared replay over the six existing local private-run artifacts.

Expected output:

`calibration/vnext/private-runs/OROTITAN_GATE18_V1_1_SHADOW_REPLAY_RESULT_001.json`

## OI-002 — Classify replay deltas by failure layer

Status: BLOCKED_BY_OI-001

Required categories:

- presentation-only fail cleared;
- downstream presentation/compliance failure;
- downstream presentation/boundary failure;
- substantive semantic failure;
- stable positive control.

## OI-003 — Decide v1.1 contract disposition

Status: BLOCKED_BY_OI-001_AND_OI-002

Possible outcomes are to be decided from replay evidence. No v1.1 contract implementation is currently authorized.

## OI-004 — Model qualification remains paused beyond current authority

Status: OPEN_GUARDED

Second Phi-4 C4 cell, model switch, and Qwen3.5 download remain unauthorized while the contract-architecture review is unresolved.
