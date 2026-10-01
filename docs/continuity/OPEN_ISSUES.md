# OroTitan VNExT — Open Issues

## OI-001 — Historical Phase C / C7
Status: COMPLETE_RETAINED_IMMUTABLE

Phase C = COMPLETE  
C7 = LOCAL_CANDIDATE_REJECTED

## OI-002 — Analytical Engine V2
Status: TARGET_ARCHITECTURE_SELECTED

No change to frozen analytical semantics.

## OI-003 — ChatGPT Operating Protocol V1.0
Status: FROZEN
Priority: CLOSED

Authoritative document:
`docs/orotitan-equity/OROTITAN_CHATGPT_OPERATING_PROTOCOL_V1_FREEZE_V1.0.md`

## OI-004 — Analytical Engine V2 data contracts
Status: TO_DESIGN
Priority: P0

Must translate the frozen methodology + ChatGPT protocol into exact machine-readable contracts without semantic duplication.

## OI-005 — Process Engine V2 state machine
Status: TO_DESIGN
Priority: P0

Must model run/stage/block/checkpoint/finalization/reopening/failure/recovery behavior.

## OI-006 — Controlled ChatGPT ↔ Supabase operations
Status: TO_DESIGN
Priority: P0

Target:
- read-only / project-scoped normal reads where practical;
- guarded RPC writes only for checkpoint/finalization/publication-related transitions;
- no ad-hoc production DML in company-analysis workflow.

## OI-007 — Workbench UI / UX
Status: TO_DESIGN
Priority: P1

Includes:
- French-first UI;
- queue / shortlist;
- current runs;
- block progress;
- blockers / next action;
- evidence browser;
- history;
- price monitoring;
- exports.

## OI-008 — Production / routing
Status: GUARDED

Model winner: NONE  
Routing freeze: NONE  
Production mutation: FALSE  
Gate 18: IN_PROGRESS_NOT_FROZEN
