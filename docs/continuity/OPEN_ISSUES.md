# OroTitan VNExT — Open Issues

## OI-001 — v1.1 validation architecture

Status: COMPLETE

The versioned v1.1 contract is implemented and CI-verified.

## OI-002 — Phi-4 C4 resumption decision

Status: COMPLETE

One bounded Constellation Software cell is selected as the next discriminator.

## OI-003 — Phi-4 Constellation v1.1 inference authorization

Status: OPEN_DECISION_REQUIRED  
Priority: P0

The guarded runner and authorization template are prepared.

Inference remains blocked because the authorization template is `PREPARED_NOT_AUTHORIZED`.

Required decision:
Explicitly authorize or reject exactly one frozen local inference.

## OI-004 — Post-run adjudication

Status: BLOCKED_BY_OI-003

If the run is authorized and completed, evaluate:

- raw presentation compliance;
- safe normalization count;
- v1.1 substantive deterministic semantics;
- runtime result;
- human-quality adjudication.

## OI-005 — Model winner and routing

Status: OPEN_GUARDED

No winner is selected and routing remains unfrozen.
