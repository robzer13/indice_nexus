# OroTitan VNExT — Open Issues

## OI-001 — v1.1 validation architecture

Status: COMPLETE

The versioned v1.1 contract is implemented and CI-verified.

## OI-002 — Phi-4 C4 resumption decision

Status: COMPLETE

Constellation Software / SERIAL_ACQUIRER is selected as the next bounded discriminator.

## OI-003 — Phi-4 Constellation v1.1 inference authorization

Status: COMPLETE

Exactly one frozen local inference is explicitly authorized.

## OI-004 — Execute the authorized local inference

Status: READY_LOCAL_EXECUTION  
Priority: P0

Run exactly one invocation of the guarded runner with the exact authorization ID.

No automatic retry is authorized if execution fails or times out.

## OI-005 — Post-run engineering and human adjudication

Status: BLOCKED_BY_OI-004

After execution, inspect:

- runtime status;
- raw schema status;
- raw presentation compliance;
- v1.1 safe normalization count;
- v1.1 substantive deterministic semantic result;
- output budget / finish reason;
- human-quality criteria.

## OI-006 — Model winner and routing

Status: OPEN_GUARDED

No model winner is selected and routing remains unfrozen.
