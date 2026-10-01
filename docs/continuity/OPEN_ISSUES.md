# OroTitan VNExT — Open Issues

## OI-001 — Phase C local-model campaign
Status: COMPLETE

C7:
`LOCAL_CANDIDATE_REJECTED`

Current local production candidate:
NONE

## OI-002 — Interim post-C7 operating strategy
Status: SELECTED_PRE_IMPLEMENTATION
Priority: P0

Strategy:
`ASSISTED_HUMAN_IN_THE_LOOP_ANALYSIS_BRIDGE`

Goal:
simplify execution and increase analytical quality without forcing weak local inference.

## OI-003 — Assisted task bundle contract
Status: NOT_IMPLEMENTED
Priority: P0

Required outputs:
- `TASK_PACKET.json`;
- `PROMPT.md`;
- `EXPECTED_OUTPUT_SCHEMA.json`;
- `CONTROL_CARD.md`.

## OI-004 — Assisted prepare/import tooling
Status: NOT_IMPLEMENTED
Priority: P0

Prepare target:
`orotitan assist prepare <company> <module>`

Import target:
`orotitan assist import <result.json>`

Import must fail closed on identity, schema, reference or deterministic-semantic defects.

## OI-005 — Repair loop
Status: NOT_IMPLEMENTED

Generate a minimal repair request from validator paths/codes without changing the evidence packet or historical output.

## OI-006 — First pilot
Status: NOT_STARTED

Module:
`MOAT_EVIDENCE_AUDIT_ASSISTED_V0_3`

Measure:
operator time, manual actions, schema pass rate, invalid-reference rate, semantic defects, repair count and human-quality defects.

## OI-007 — Production / routing
Status: GUARDED

Model winner:
NONE

Routing freeze:
NONE

Production mutation:
FALSE

Gate 18:
IN_PROGRESS_NOT_FROZEN
