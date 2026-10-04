# OROTITAN_PRE_CERTIFICATION_CHECKLIST_SPEC_V4_DRAFT_V0.1

Status: DRAFT — NON AUTHORITATIVE
Methodology change: YES
Production activation: NOT AUTHORIZED
Frozen V2 / V3 contracts: UNCHANGED
Target successor family: V4
Registry stage code: DEEP_DIVE unchanged
Analytical phase: PHASE 2.5 — PRE_CERTIFICATION_CHECKLIST

## 0. Purpose

Target analytical pipeline:

RESEARCH → DEEP DIVE / FUNDAMENTALS → VALUATION → PRE-CERTIFICATION CHECKLIST → CERTIFICATION / VALIDATION → INTEGRATION → PUBLICATION.

The checklist does not redo the Deep Dive. Its purpose is to detect what Research + Deep Dive + Valuation may have missed before the investment conclusion is allowed to enter Certification. It is the final INVESTMENT DECISION CHALLENGE.

## 1. Cognitive boundary

The checklist MUST NOT become a second Moat analysis, Runway analysis, ROIC / ROIIC analysis, FCF analysis, Management analysis, Risk block, Valuation model or broad evidence-gathering stage.

It may challenge those outputs. It may not silently replace them.

If the challenge reveals a material defect, the exact upstream scope is reopened, repaired or updated, dependent Valuation is rerun where required, and the checklist is rerun.

## 2. Admission

The checklist may begin only when the current FUNDAMENTALS_LOCK, VALUATION_LOCK and VALUATION_ARTIFACT are exact, current and hash-resolvable; the authoritative Evidence / Conflict / Calculation / Assumption lineage is exact; RUN_ID, DATA_CUTOFF and contract pins are unchanged; DEEP_DIVE remains IN_PROGRESS; and READY_FOR_PRE_CERTIFICATION_CHECKLIST = YES.

Valuation no longer hands directly to Certification under this successor.

## 3. Internal gates

Successor Valuation emits READY_FOR_PRE_CERTIFICATION_CHECKLIST = YES.

The checklist owns READY_FOR_CERTIFICATION = YES / NO.

Certification may start only from a passing current checklist.

## 4. Question acceptance rule

A generated question is executed only if it contributes a materially new challenge.

At least one condition must hold: it reveals a cross-block contradiction, hidden assumption, critical dependency, blind spot, materially different world, second/third-order effect, valuation fragility, risk/reward asymmetry, potential decision change, or a reason to reopen upstream work.

Generated candidates that fail the test are recorded as DROP_DUPLICATE, DROP_ALREADY_RESOLVED, DROP_NON_MATERIAL or DROP_NOT_COMPANY_RELEVANT. Dropped questions do not count toward executed depth.

## 5. Adaptive depth

100 executed questions is a minimum depth, not a target, maximum or stopping rule.

Normal completion requires TOTAL_QUESTIONS_EXECUTED >= 100 and NO_ADDITIONAL_MATERIAL_DECISION_USEFUL_QUESTION_IDENTIFIED.

The final saturation record should show mandatory family coverage, company-specific coverage, disposition of all material concerns, and at least two final independent question-generation passes producing zero newly accepted material questions.

Question count is descriptive only and has no scoring authority.

## 6. Mandatory challenge families

A. CROSS_BLOCK_CONSISTENCY: challenge whether Fundamentals, risk and Valuation assumptions are jointly consistent. This is adversarial challenge, not the formal Certification-owned CROSS_BLOCK_RECONCILIATION.

B. GREAT_INVESTOR_DECISION_THINKING: inversion, permanent capital loss, margin of safety, opportunity cost, second-level thinking, circle of competence, behavioral risk, asymmetry, base rates and what-must-be-true. Generic slogans are insufficient; each question must bind to a company-specific assumption, exposure or valuation state.

C. WORLD_IN_MOTION: challenge geopolitics, wars, sanctions, elections, public policy, EU, tax, inflation/deflation, rates, credit, FX, protectionism, tariffs, industrial policy, regulation, antitrust, demographics, social change, climate and physical events. Use EVENT → EXPOSURE → TRANSMISSION MECHANISM → ECONOMIC IMPACT → VALUATION / THESIS IMPACT → MITIGATION. Do not predict the event. Do not introduce actual post-cutoff facts.

D. OPERATIONAL_REALITY_SUPPLY_CHAIN: single points of failure, suppliers, Tier-2/Tier-3 dependencies, bottlenecks, capacity, qualification lead times, transport, ports/straits/corridors, raw materials, energy, water, infrastructure, inventories, scarce talent, critical sites and geographic concentration. Financial projections must also be physically feasible.

E. TECHNOLOGY_AI: AI or other technology as threat, opportunity, commoditization, lower barrier to entry, disintermediation, pricing-model threat, seat/headcount risk, client internalization or moat amplifier. Also test the inverse world where expected technological impact is much weaker.

F. SECOND_ORDER_EFFECTS: for material shocks explicitly test first-order, second-order and, where material, third-order effects.

G. COUNTERFACTUALS: excellent company but mediocre investment; acceptable growth with much higher capital needs; higher margins but lower ROIC; moat survives but market size disappoints; earnings grow but per-share value stagnates; fundamentally sound business with structurally lower multiple; no catastrophe but 10–15 years of mediocre shareholder returns.

H. COMPANY_SPECIFIC_CHALLENGE: questions derived from the actual sector, geography, business model, suppliers, customers, pricing, technologies, regulation, capital structure, vulnerabilities and valuation. If the company name can be replaced by almost any company without materially changing the question, the question is probably insufficiently specific.

These families are minimum coverage, not a closed maximum list.

## 7. Narrow research rule

The checklist is not a broad research stage.

Allowed: targeted verification triggered by an accepted challenge question, checking a specifically identified dependency/contradiction, and adding cutoff-compliant evidence to the existing authoritative Evidence Ledger lineage.

Not allowed: broad exploratory research, rebuilding an upstream block inside the checklist, parallel evidence bases or silent resolution of material contradictions.

Material upstream work produces REOPEN and returns ownership to the exact upstream block.

## 8. Per-question record

Each executed question should preserve: QUESTION_ID, QUESTION_FAMILY, optional subfamily, QUESTION_TEXT, COMPANY_SPECIFIC, ORIGIN_TRIGGER, WHY_MATERIAL, NOVELTY_BASIS, ANSWER_OR_JUDGMENT, STATUS, EVIDENCE_REFERENCES, AFFECTED_BLOCKS, REOPEN_TARGET if applicable, DECISION_IMPACT, MITIGATION_OR_RESOLUTION and first/second/third-order effects where relevant.

ORIGIN_TRIGGER should point to the exact analytical assumption, dependency, conflict, valuation input or exposure that motivated the question.

## 9. Status semantics

PASS: no new material problem.

CONCERN: a real material issue is identified but is understood, bounded/mitigated or already incorporated, and does not require reopening. It must be propagated to Certification.

REOPEN: unresolved issue requires modification or re-execution of upstream work. READY_FOR_CERTIFICATION = NO.

FAIL: the current investment conclusion is no longer defensible. READY_FOR_CERTIFICATION = NO. FAIL applies to the current conclusion; a materially new upstream conclusion may later be challenged again.

## 10. Non-compensatory aggregate state

No CHECKLIST_SCORE, weighted score, average status or pass-rate decision score is permitted.

If any unresolved FAIL exists: CHECKLIST_STATUS = FAIL and READY_FOR_CERTIFICATION = NO.
Else if any unresolved REOPEN exists: CHECKLIST_STATUS = REOPEN and READY_FOR_CERTIFICATION = NO.
Else if one or more CONCERN exists: CHECKLIST_STATUS = PASS_WITH_CONCERNS and READY_FOR_CERTIFICATION = YES.
Else: CHECKLIST_STATUS = PASS and READY_FOR_CERTIFICATION = YES.

Counts never compensate for one material negative discovery.

## 11. Reopening loop

VALUATION_LOCK vN → CHECKLIST vN → REOPEN → preserve prior artifacts → reopen exact scope → create new affected versions → invalidate dependent Valuation when required → VALUATION_LOCK vN+1 → CHECKLIST vN+1 → PASS / PASS_WITH_CONCERNS → CERTIFICATION.

During this normal loop the Registry stage remains DEEP_DIVE, lifecycle remains IN_PROGRESS unless a true pause/blocker applies, active Stage Manifest remains CHECKPOINT, and READY_FOR_INTEGRATION is not YES.

A full Registry stage revision is required only if a previously finalized Deep Dive is later reopened.

## 12. Required authoritative artifacts

PRE_CERTIFICATION_QUESTION_LEDGER: all executed questions, judgments, statuses, evidence references, affected blocks and reopen targets.

PRE_CERTIFICATION_CHECKLIST_REPORT: decision-level summary containing COMPANY, RUN_ID, DATA_CUTOFF, CHECKLIST_VERSION, CHECKLIST_ITERATION, candidate/executed/drop counts, PASS/CONCERN/REOPEN/FAIL counts, MATERIAL_CONCERNS, BLIND_SPOTS_IDENTIFIED, COMPANY_SPECIFIC_QUESTIONS, REOPEN_REQUIRED, FAIL_REASONS, AFFECTED_ANALYTICAL_BLOCKS, SATURATION_RECORD, CHECKLIST_STATUS and READY_FOR_CERTIFICATION.

Counts are descriptive metadata only.

## 13. Saturation record

Retain MINIMUM_DEPTH_SATISFIED, MANDATORY_FAMILIES_COVERED, COMPANY_SPECIFIC_COVERAGE, ALL_MATERIAL_CONCERNS_DISPOSITIONED, FINAL_ZERO_YIELD_PASSES and STOP_REASON.

Normal stop reason: NO_ADDITIONAL_MATERIAL_DECISION_USEFUL_QUESTION_IDENTIFIED.

No hard maximum exists.

## 14. Certification boundary

CHECKLIST PASS is not CERTIFIED and is not VALIDATED.

Checklist means the current investment conclusion survived the final adversarial investment-decision challenge.

Certification remains responsible for evidence sufficiency, calculation reproducibility, methodology compliance, traceability, formal cross-block reconciliation, score permission, deterministic scoring, terminal gate and final readiness / next action.

## 15. Certification admission

Certification requires exact current FUNDAMENTALS_LOCK, VALUATION_LOCK, PRE_CERTIFICATION_QUESTION_LEDGER and PRE_CERTIFICATION_CHECKLIST_REPORT; CHECKLIST_STATUS in PASS or PASS_WITH_CONCERNS; READY_FOR_CERTIFICATION = YES; no unresolved REOPEN/FAIL; and exact current hashes/lineage.

Material concerns must propagate to Certification limitations and final thesis where relevant.

## 16. Integration boundary

Integration remains technical/canonical. It must not generate or answer checklist questions, resolve concerns, reopen analysis, reinterpret statuses, repair failure, or infer pass from counts.

Integration verifies that the exact certified Deep Dive lineage contains the required current checklist artifacts.

Checklist absent, stale, superseded, REOPEN or FAIL blocks Integration admission.

## 17. Historical firewall

Frozen V2/V3 runs keep their original Contract Sets and are never retrofitted silently.

A methodology replay uses a controlled successor run.

## 18. Successor engine identity

Recommended successor Contract Set adds a dedicated pre_certification_checklist pin with immutable locator and SHA-256. This changes contract_set_sha256 and therefore engine provenance.

## 19. Acceptance tests before freeze

PC01 Valuation complete but checklist absent → Certification blocked.
PC02 fewer than 100 executed questions → normal completion blocked.
PC03 100 questions but material questions remain → completion blocked.
PC04 duplicates/already-resolved items do not count.
PC05 PASS → READY_FOR_CERTIFICATION YES.
PC06 PASS_WITH_CONCERNS → YES.
PC07 unresolved REOPEN → blocked.
PC08 unresolved FAIL → blocked.
PC09 reopening Fundamentals invalidates dependent Valuation when materially affected.
PC10 reopening Valuation alone leaves Fundamentals Lock intact.
PC11 stale checklist version cannot certify against newer Valuation Lock.
PC12 post-cutoff factual contamination → rejected.
PC13 hypothetical future scenario without contamination → allowed.
PC14 checklist does not calculate OQS/OVS/Investment Score.
PC15 Integration cannot manufacture or repair checklist outputs.
PC16 ledger/report exact and hash-resolvable.
PC17 PASS count cannot compensate for one FAIL.
PC18 company-specific coverage required.
PC19 no artificial max count.
PC20 saturation rationale required.
PC21 V2/V3 historical run remains valid without checklist.
PC22 successor fingerprint differs.
PC23 Certification formal reconciliation remains distinct.
PC24 broad research routes upstream instead of expanding checklist.

## 20. Freeze rule

This draft has no authority until explicit user methodology approval, V4 process/pilotage/deep-dive successors, artifact schemas, pin pack, regression/runtime tests and formal freeze exist.

Until then CURRENT FROZEN V2/V3 and production pipeline remain unchanged.
