# OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3 — FREEZE V3.0

**Status:** FROZEN — V3.0
**Base incorporated by reference:** OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V2_FREEZE_V2.0 @ SHA256 1b2b526b2199a4a487b75e0f0ea496180964c4d5949f792275caf5379b67d97a
**Depends on:** OROTITAN_EXECUTION_PROCESS_V3_FREEZE_V3.0
**Methodology authority:** NONE

All V2 routing rules remain unless superseded here.

## 1. Denominator admission routing

For VALUATION admission Pilotage verifies the exact pinned denominator authority and authoritative denominator state.

VALuation may start only when:
- ECONOMIC_SHARE_COUNT representation = EXACT and valuation_admission = YES; or
- representation = BOUNDED, BOUND_COMPLETENESS = COMPLETE, UNBOUNDED_MOVEMENT_CLASSES = 0, valuation_admission = YES.

UNKNOWN, malformed, contradictory or unpinned denominator state blocks Valuation.

Pilotage never constructs or narrows a denominator bound.

## 2. Historical Contract Set firewall

Existing runs retain immutable contract_pins, contract_set_sha256 and DATA_CUTOFF.

When the new methodology is required for a run created under an earlier Contract Set:
- CURRENT_RUN_CONTRACT_REBIND = NO;
- create a controlled successor run only after the V3 Contract Set is production-admissible;
- parent_run_id must equal the historical run ID.

Pure methodology replay retains the historical DATA_CUTOFF. Later evidence requires explicit METHODOLOGY_REPLAY_PLUS_INFORMATION_REFRESH classification.

## 3. Handoffs

V2 Research -> Fundamentals and checkpoint semantics remain unchanged.

Valuation -> Certification still requires an exact valid VALUATION_LOCK and READY_FOR_CERTIFICATION = YES.

No bounded denominator may bypass Certification, Integration or publication controls.

## 4. No side effects

This contract authorizes no live run creation, current-run mutation, publication or canonical pointer movement by itself.
