# OroTitan VNExT — Open Issues

## OI-001 — Historical Phase C / C7
Status: COMPLETE_RETAINED_IMMUTABLE

Phase C = COMPLETE  
C7 = LOCAL_CANDIDATE_REJECTED

## OI-002 — Analytical Engine V2
Status: TARGET_ARCHITECTURE_SELECTED

No change to frozen analytical semantics.

## OI-003 — ChatGPT Operating Protocol V0.1
Status: DESIGN_CANDIDATE_READY_FOR_USER_REVIEW
Priority: P0

Candidate:
`docs/orotitan-equity/OROTITAN_CHATGPT_OPERATING_PROTOCOL_V0.1.md`

Must not be frozen until user approval.

## OI-004 — Supabase ChatGPT access hardening
Status: DESIGN_REQUIREMENT_NOT_IMPLEMENTED
Priority: P0

Target:
- project-scoped routine reads;
- read-only behavior for normal LOAD / STATUS / Research retrieval where practical;
- bounded write window for CHECKPOINT / SAVE / FINALIZE;
- guarded existing OroTitan RPCs only for normal analytical writes;
- no ad-hoc DML in company-analysis workflow.

## OI-005 — Detailed implementation design
Status: WAITING_FOR_PROTOCOL_APPROVAL

After protocol approval:
1. Analytical Engine V2 data contracts
2. Process Engine V2 state machine
3. ChatGPT/Supabase operation payload contracts
4. Workbench information architecture
5. UI design system
6. implementation backlog

## OI-006 — Production / routing
Status: GUARDED

Model winner: NONE  
Routing freeze: NONE  
Production mutation: FALSE  
Gate 18: IN_PROGRESS_NOT_FROZEN
