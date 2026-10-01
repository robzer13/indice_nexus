# OroTitan VNExT — Open Issues

## OI-001 — ChatGPT Operating Protocol V1.0
Status: FROZEN
Priority: CLOSED

## OI-002 — Protocol regression battery
Status: BASELINE_EXECUTED
Priority: CLOSED_FOR_CURRENT_PHASE

Result:

`FOUNDATION_PASS_WITH_IMPLEMENTATION_GAPS`

25 scenarios mapped.  
15 require implementation closure before vertical slice.

## OI-003 — Analytical Engine V2 Data Contracts
Status: TO_DESIGN
Priority: P0

Must close at minimum:
- post-cutoff source validation;
- evidence / conflict structures;
- Evidence-ID validation;
- serial-acquirer / capital-allocation structures;
- cross-block references needed by revalidation.

## OI-004 — Process Engine V2
Status: TO_DESIGN
Priority: P0

Must close at minimum:
- material-change revalidation gate;
- sector-overlay validation;
- block-level gaps / NOT_ASSESSABLE;
- dependency reopening;
- price-only / routine / full refresh routing.

## OI-005 — ChatGPT ↔ Supabase bridge
Status: TO_DESIGN
Priority: P0

Must close at minimum:
- LOAD resolver;
- layered context assembler;
- fail-closed connector behavior;
- SAVE orchestration;
- finalization eligibility;
- bounded guarded-RPC invocation.

## OI-006 — Supabase security hardening
Status: TRACKED
Priority: P1 BEFORE FINAL PRODUCTION

- document intended service-only RLS posture;
- harden 3 mutable-search-path legacy functions.

## OI-007 — Workbench UI / UX
Status: WAITING_FOR_FOUNDATION
Priority: P1

Final vitrine is French-first.

Implementation should start only after the functional vertical slice validates the underlying data/state model.

## OI-008 — Production / routing
Status: GUARDED

Model winner: NONE  
Routing freeze: NONE  
Production mutation: FALSE  
Gate 18: IN_PROGRESS_NOT_FROZEN
