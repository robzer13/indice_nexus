# OROTITAN_PRE_CERTIFICATION_ARCHITECTURE_IMPACT_V4_DRAFT_V0.1

Status: DRAFT — ARCHITECTURE IMPACT ONLY
Runtime mutation: NONE
Database migration authorized: NO
Frozen contracts modified: NO

## 1. Architectural decision

Recommended Registry stages remain RESEARCH → DEEP_DIVE → INTEGRATION.

Deep Dive successor phases become FUNDAMENTALS → VALUATION → PRE_CERTIFICATION_CHECKLIST → CERTIFICATION_RECONCILIATION.

This preserves the requested analytical pipeline without creating an asymmetric Registry stage between two phases that are already internal to Deep Dive.

## 2. Why no new stage_code

Valuation is currently an internal Deep Dive phase. Certification is currently an internal Deep Dive phase. The existing checkpoint protocol already persists phase handoffs, and the checklist's natural REOPEN targets are analytical artifacts inside Deep Dive.

A physical PRE_CERTIFICATION stage would require splitting Valuation and Certification into separate Registry stages as well; otherwise the topology would be inconsistent.

Therefore the recommended V4 change is a mandatory first-class internal phase 2.5, not a fourth Registry stage.

## 3. Frozen conflict

Current V2/V3: VALUATION_LOCK → READY_FOR_CERTIFICATION → CERTIFICATION.

V4 target: VALUATION_LOCK → READY_FOR_PRE_CERTIFICATION_CHECKLIST → CHECKLIST → READY_FOR_CERTIFICATION → CERTIFICATION.

This is an explicit methodology change and cannot patch V2/V3 silently.

## 4. Contract impact

Successor versions are required for Execution Process, Pilotage, Deep Dive, Master Prompt/execution sequencing where relevant, Contract Pin Pack and Runtime Bootstrap.

Research Stage may remain unchanged if its semantics are unaffected.

Integration remains analytically unchanged but its successor admission validation must verify checklist lineage.

Recommended dedicated pin: pre_certification_checklist.

## 5. Deep Dive contract impact

Phase 1 Fundamentals: inherited.
Phase 2 Valuation: outgoing gate becomes READY_FOR_PRE_CERTIFICATION_CHECKLIST.
Phase 2.5 Checklist: produces PRE_CERTIFICATION_QUESTION_LEDGER and PRE_CERTIFICATION_CHECKLIST_REPORT.
Phase 3 Certification: requires passing current checklist.

Certification remains the only normal phase that finalizes Deep Dive.

## 6. Registry impact

No new stage_code recommended. Existing stage constraints therefore need no topological migration.

Existing contract_status_code can represent internal phase states such as VALUATION_LOCKED_READY_FOR_PRE_CERTIFICATION, PRE_CERTIFICATION_IN_PROGRESS, PRE_CERTIFICATION_REOPEN_REQUIRED, PRE_CERTIFICATION_FAILED, PRE_CERTIFICATION_READY_FOR_CERTIFICATION and CERTIFICATION_IN_PROGRESS.

Deep Dive handoff remains READY_FOR_INTEGRATION.

## 7. Manifest impact

Checkpoint sequence becomes Fundamentals CHECKPOINT → Valuation CHECKPOINT → Pre-Certification CHECKPOINT → final Deep Dive Manifest after Certification.

The checklist checkpoint includes exact refs to current locks, ledger/report and required evidence/calculation lineage.

CHECKPOINT never admits Integration.

## 8. Required-output enforcement

Current Registry finalization requires the historical Deep Dive artifact family but does not require checklist artifacts.

V4 runtime should require PRE_CERTIFICATION_QUESTION_LEDGER and PRE_CERTIFICATION_CHECKLIST_REPORT only when the pinned Deep Dive stage contract is V4+.

Do not globally require these artifacts for V2/V3 historical runs.

The validator must therefore be contract-version-aware.

## 9. RPC impact

start_orotitan_stage: no new branch required under this architecture.
checkpoint_orotitan_stage: existing mechanism can persist the additional Deep Dive checkpoint.
finalize_orotitan_stage: topology unchanged, but successor final-manifest validation must enforce checklist artifacts and passing lineage.
reopen_orotitan_stage: not used for a normal checklist REOPEN while Deep Dive is still IN_PROGRESS. A checklist REOPEN is an internal analytical-scope loop. Full stage reopen remains for already-finalized Deep Dive only.

## 10. Reopen targets

Recommended controlled targets: RESEARCH where evidence sufficiency itself failed; BUSINESS_MODEL; MOAT; RUNWAY; RETURN_QUALITY; FCF_FORENSIC; CAPITAL_ALLOCATION; MANAGEMENT_GOVERNANCE; RISK_RESILIENCE; VALUATION.

Reopen exact material scope only. Preserve prior versions. Invalidate downstream artifacts when dependencies change. Rerun Valuation when its inputs are affected. Rerun checklist after every material upstream revision.

## 11. Checklist state machine

NOT_STARTED → IN_PROGRESS → PASS / PASS_WITH_CONCERNS / REOPEN / FAIL.

PASS and PASS_WITH_CONCERNS produce READY_FOR_CERTIFICATION YES.
REOPEN and FAIL produce READY_FOR_CERTIFICATION NO.
Later repaired iterations supersede but never erase prior versions.

## 12. Artifact lineage

PRE_CERTIFICATION_QUESTION_LEDGER consumes/derives from current FUNDAMENTALS_LOCK, VALUATION_LOCK and relevant ledgers.
PRE_CERTIFICATION_CHECKLIST_REPORT consumes the question ledger.
CERTIFICATION_ARTIFACT consumes the current passing report/ledger plus current locks and ledgers.

Only Registry-supported edge types may be used unless relation vocabulary is explicitly versioned.

## 13. Integration impact

Successor Integration preflight verifies: exact final Deep Dive manifest; exact current checklist report and question ledger; PASS/PASS_WITH_CONCERNS; READY_FOR_CERTIFICATION YES; Certification references the same checklist version/hash; and no later upstream artifact invalidates the checklist.

Integration does not ask questions, reinterpret answers, change status or repair failures.

## 14. Canonical snapshot impact

No checklist score enters canonical scoring.

Optional audit projection may expose checklist_version, checklist_status, question_count, concern_count, data_cutoff and artifact_ref.

Full question bodies remain authoritative artifacts, not duplicated into the canonical snapshot.

## 15. DATA_CUTOFF

The checklist shares the run's immutable DATA_CUTOFF.

Hypothetical future worlds are allowed. Actual post-cutoff evidence is not. Exposure/transmission evidence must be cutoff-compliant. Later facts require governed refresh/successor routing.

## 16. Contract pin / fingerprint

A new pre_certification_checklist pin is recommended. The current pin-completeness guard requires the historical 13 pins but does not reject extra pins, so V4 can add the checklist pin while V2/V3 remain valid.

The contract-set hash includes all pins, so V4 receives a distinct engine fingerprint automatically.

## 17. Later implementation areas

V4 Process; V4 Pilotage; V4 Deep Dive; Checklist methodology contract; V4 pin pack; V4 runtime bootstrap; checklist artifact schemas; V4-aware Deep Dive final output validator; V4-aware Integration preflight; regression tests; audit UI; and engine provenance activation only after formal production authorization.

No production mutation is authorized by this draft.

## 18. Decision

ADD PRE_CERTIFICATION AS MANDATORY DEEP_DIVE PHASE 2.5.
DO NOT ADD A NEW REGISTRY STAGE_CODE.
DO NOT MODIFY V2/V3 FROZEN CONTRACTS.
BUILD AS V4 SUCCESSOR.
PERSIST QUESTION LEDGER + CHECKLIST REPORT.
USE NON-COMPENSATORY PASS / CONCERN / REOPEN / FAIL.
BLOCK CERTIFICATION ON REOPEN / FAIL.
KEEP INTEGRATION NON-ANALYTICAL.
