# OroTitan VNExT — Open Issues

## OI-001 — Execute v1.1 shadow replay

Status: COMPLETE  
Inference required: NO

Result source:

`calibration/vnext/private-runs/OROTITAN_GATE18_V1_1_SHADOW_REPLAY_RESULT_001.json` on the user's local machine.

Six cases completed with no new inference and no source mutation.

## OI-002 — Classify replay deltas by failure layer

Status: COMPLETE

Result:

- presentation-only fail cleared: 3 cases;
- retained substantive semantic failure: 2 cases;
- stable positive control: 1 case;
- newly revealed downstream failure after normalization: 0 cases.

Important:
RATIONAL also has one separate presentation-boundary saturation path that safe normalization intentionally does not touch; its substantive conflict-grounding failure remains independently present.

## OI-003 — Decide v1.1 contract disposition

Status: OPEN_DECISION_REQUIRED  
Priority: P0

The shadow replay supports the two-layer architecture hypothesis for the tested contract-architecture purpose.

No v1.1 contract change is currently authorized.

Required decision:
Authorize or reject a versioned v1.1 contract change. If authorized, specify and implement it separately with historical v1.0 immutability preserved.

## OI-004 — Model qualification remains paused beyond current authority

Status: OPEN_GUARDED

Second Phi-4 C4 cell, model switch, and Qwen3.5 download remain unauthorized until the contract-architecture disposition is resolved.

## OI-005 — Reconcile authoritative Phase C entry after disposition

Status: BLOCKED_BY_OI-003

`OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json` still names the now-completed shadow replay as `next_action`.

Do not mutate that authoritative Gate-state artifact merely from the continuity layer. Reconcile it only through the proper post-replay decision authority.
