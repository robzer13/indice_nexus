# OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V4_DRAFT_V0.1

Status: DRAFT — NON AUTHORITATIVE
Proposed base: OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3_FREEZE_V3.0
Registry stage code: DEEP_DIVE
Methodology delta: mandatory PRE_CERTIFICATION_CHECKLIST phase
Production activation: NO

This draft defines only the sequencing delta. All V3 rules remain incorporated by reference unless explicitly superseded below.

## 1. Proposed phase order

PHASE 1 = FUNDAMENTALS.
PHASE 2 = VALUATION.
PHASE 2.5 = PRE_CERTIFICATION_CHECKLIST.
PHASE 3 = CERTIFICATION_RECONCILIATION.

Only Phase 3 may normally finalize the DEEP_DIVE Registry stage.

## 2. Phase 2 outgoing gate

VALUATION_LOCK no longer emits direct certification readiness.

Successor output is READY_FOR_PRE_CERTIFICATION_CHECKLIST = YES.

Valuation routes to Phase 2.5.

## 3. Phase 2.5 admission

Requires exact current FUNDAMENTALS_LOCK, VALUATION_LOCK, VALUATION_ARTIFACT, READY_FOR_PRE_CERTIFICATION_CHECKLIST = YES, exact current ledger lineage, and unchanged RUN_ID / DATA_CUTOFF / contract pins.

## 4. Phase 2.5 execution

Execute the pinned PRE_CERTIFICATION_CHECKLIST methodology.

Persist PRE_CERTIFICATION_QUESTION_LEDGER and PRE_CERTIFICATION_CHECKLIST_REPORT.

Produce CHECKLIST_STATUS and READY_FOR_CERTIFICATION.

No OQS, OVS or Investment Score may be produced by this phase.

## 5. Phase 2.5 checkpoint

Every durable iteration persists exact artifacts, verifies hashes, registers artifacts plus a CHECKPOINT Deep Dive Stage Manifest, keeps DEEP_DIVE lifecycle IN_PROGRESS and keeps READY_FOR_INTEGRATION not YES.

REOPEN sets READY_FOR_CERTIFICATION NO and routes to exact analytical owner before rerunning checklist.

FAIL sets READY_FOR_CERTIFICATION NO and blocks Certification for the current conclusion.

PASS or PASS_WITH_CONCERNS sets READY_FOR_CERTIFICATION YES and produces the exact Certification bootstrap.

## 6. Phase 3 admission delta

Certification requires current question ledger, current checklist report, CHECKLIST_STATUS PASS or PASS_WITH_CONCERNS, READY_FOR_CERTIFICATION YES, no unresolved REOPEN/FAIL and proof that the checklist consumed the current Valuation Lock.

Certification may not bypass checklist failure.

## 7. Certification distinction

Certification remains control/reconciliation and does not rerun the checklist.

Formal CROSS_BLOCK_RECONCILIATION remains Certification-owned.

Material checklist concerns propagate to Certification limitations, final structured thesis, invalidation triggers and readiness/next action where relevant.

## 8. Final artifact delta

Successor final Deep Dive lineage adds PRE_CERTIFICATION_QUESTION_LEDGER and PRE_CERTIFICATION_CHECKLIST_REPORT. All inherited V3 outputs remain.

## 9. Finalization delta

READY_FOR_INTEGRATION may be YES only when checklist artifacts are exact/current, status is PASS or PASS_WITH_CONCERNS, Certification executed, score permission obeyed, deterministic outputs reconciled and all inherited completion rules pass.

## 10. Historical firewall

No V2/V3 run is rebound to this draft. No historical artifact is rewritten. This document has zero execution authority until formally frozen in a successor Contract Set.
