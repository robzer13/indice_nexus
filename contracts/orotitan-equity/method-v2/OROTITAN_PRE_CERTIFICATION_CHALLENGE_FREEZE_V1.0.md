# OroTitan Method-V2 Pre-Certification Challenge — Freeze V1.0

STATUS = FROZEN
METHODOLOGY_CHANGE = YES
PRODUCTION_ACTIVE = NO
AUTHORIZATION = METHOD-V2-AUTHORITY-CLOSURE-001
REGISTRY_STAGE_CODE = DEEP_DIVE
QUESTION_NUMERIC_QUOTA = NONE

## 1. Purpose and authority

This is the final independent adversarial investment-decision challenge between
Valuation and Certification. It supersedes direct Valuation-to-Certification
admission only for a new run explicitly bound to the exact Method-V2 analytical
manifest. It does not certify, score, publish, replace an upstream block, create
a Registry stage, or change the downstream Certification methodology.

Source design: `vnext-pre-certification-v4-draft-001` at
`2cefd74835db19bbe5f856ac2796ff732b9150a7`, architecture impact, checklist spec,
Deep Dive delta and artifact schemas. That draft is provenance, not authority.
The approved correction removes its 100-question gate, minimum-depth boolean
and count-based completion tests. The Method-V2 name is an explicit new
methodology-generation identity, unrelated to historical runtime V2/V3 labels.

## 2. Admission and cognitive boundary

Sequence: FUNDAMENTALS → VALUATION → PRE_CERTIFICATION_CHALLENGE → CERTIFICATION.
Admission requires current hash-resolvable FUNDAMENTALS_LOCK, VALUATION_LOCK,
VALUATION_ARTIFACT and Evidence/Conflict/Calculation/Assumption lineage; exact
RUN_ID, DATA_CUTOFF and authority identity; DEEP_DIVE IN_PROGRESS; and
READY_FOR_PRE_CERTIFICATION_CHALLENGE = YES. Valuation cannot emit direct
Certification readiness for Method-V2.

The challenger must make a distinct adversarial pass independent of the upstream
conclusion being defended. Independence means a separately recorded generation
pass with its own adversarial rationale, not merely rephrasing the upstream
author's defense; a different model/vendor is neither required nor sufficient.
It may challenge Moat, Runway, ROIC/ROIIC, cash economics, management, risk or
Valuation but must not rebuild them inside the Challenge. Targeted verification
is permitted only for an accepted material question, using the same authoritative
Evidence Ledger. Broad research or substantive repair routes to the exact owner.

## 3. Minimum coverage families

- CROSS_BLOCK_CONSISTENCY: jointly test Fundamentals, risks and Valuation
  assumptions. Formal CROSS_BLOCK_RECONCILIATION remains Certification-owned.
- GREAT_INVESTOR_DECISION_THINKING: inversion, permanent loss, margin of safety,
  opportunity cost, second-level thinking, competence, behavior, asymmetry,
  base rates and what must be true, tied to this company's decision.
- WORLD_IN_MOTION: geopolitics, public policy, macroeconomics, rates, credit, FX,
  trade, regulation, demographics and physical events. Record EVENT → EXPOSURE
  → TRANSMISSION → ECONOMIC IMPACT → VALUATION/THESIS IMPACT → MITIGATION.
- OPERATIONAL_REALITY_SUPPLY_CHAIN: critical suppliers and sites, second/third
  tier dependencies, capacity, qualification, transport, energy, water, raw
  materials, inventories and talent. Test physical feasibility of projections.
- TECHNOLOGY_AI: disruption, commoditization, pricing, internalization,
  disintermediation or moat reinforcement; also test a weaker-impact world.
- SECOND_ORDER_EFFECTS: first, second and materially relevant third-order effects.
- COUNTERFACTUALS: excellent company/mediocre investment, growth/capital burden,
  margins/ROIC divergence, smaller market, per-share stagnation, lower multiple,
  and prolonged mediocre returns without catastrophe.
- COMPANY_SPECIFIC_CHALLENGE: actual sector, geography, business model,
  counterparties, pricing, technology, regulation, capital structure and valuation.

These are minimum coverage families, not a closed universe or score components.
Coverage is justified by company-specific questions or an evidenced explanation
of why a family adds no material challenge. Generic slogans do not establish it.

## 4. Adaptive generation, materiality and modes

Accept a question only if it is materially new and decision-useful: contradiction,
hidden assumption, critical dependency, blind spot, distinct world, higher-order
effect, valuation fragility, asymmetry, possible decision change or upstream
reopen. Record rejected candidates as DROP_DUPLICATE, DROP_ALREADY_RESOLVED,
DROP_NON_MATERIAL or DROP_NOT_COMPANY_RELEVANT, with reasons. Counts are telemetry.

FULL is required for initial Challenge and when changes invalidate broad coverage.
DELTA is permitted only with an exact prior passing ledger/report and a documented
dependency impact map proving which coverage remains valid. Revalidate inherited
coverage against current locks; challenge every affected scope. Uncertain or
materially wider impact requires FULL. DELTA cannot waive any family, concern
disposition or saturation requirement. The complete current conclusion, including
carried-forward concerns, remains subject to the final independent passes.

## 5. Sufficient breadth and material saturation

No universal minimum or maximum number of candidate or executed questions exists.
No count, elapsed time, token budget, pass rate or nominal completion percentage
can authorize stopping. Continue while materially decision-useful new questions
remain. Normal completion requires all of:

1. Adequately justified coverage of every mandatory family.
2. Adequate company-specific challenge.
3. All material concerns dispositioned; unresolved repair or failure blocks.
4. Independent final generation passes over the whole current conclusion yielding
   no new material decision-useful challenge, with provenance and rationale.

Retain at least two distinct final zero-yield pass records as independence evidence,
inherited from the reviewed draft; this is not a question-count quota. A positive
yield invalidates the stopping claim: execute/disposition those questions and
perform new final passes after repairs. A resource stop is PAUSED_MISSING_INPUT,
never PASS. Stop reason is NO_ADDITIONAL_MATERIAL_DECISION_USEFUL_QUESTION_IDENTIFIED.

## 6. Status and non-compensatory aggregation

Question status: PASS (no material problem), CONCERN (real but understood,
bounded/mitigated or incorporated; no repair required), REOPEN (upstream repair
required), FAIL (current investment conclusion indefensible).

Aggregate precedence: any unresolved FAIL → FAIL; otherwise any unresolved REOPEN
→ REOPEN; otherwise any CONCERN → PASS_WITH_CONCERNS; otherwise PASS. Positive
findings never compensate for a negative finding. No Challenge score, weights,
average, caps, or numerical pass-rate admission is permitted.

PASS and PASS_WITH_CONCERNS admit Certification only after breadth, saturation
and current lineage checks pass. REOPEN/FAIL always block. The report's aggregate
and counts must reconcile exactly to the effective ledger, including carried
questions in DELTA mode. Earlier resolved versions remain in immutable history.

## 7. Controlled reopen and lineage

Record exact affected block, reason, dependency cone and required repair. Supported
targets include Research for evidence insufficiency and the actual upstream
Fundamentals/Valuation owners. Never invent a second evidence base. Preserve old
artifact versions; create new versions and explicit CONSUMES/DERIVED_FROM/
SUPERSEDES/REVALIDATES edges using supported Registry vocabulary. Reopen of
Fundamentals invalidates dependent Valuation where affected; a Valuation-only
repair does not rewrite the Fundamentals Lock. Material repair always invalidates
the prior Challenge and requires rerun against current locks.

The ordinary internal loop keeps DEEP_DIVE IN_PROGRESS, manifest CHECKPOINT,
READY_FOR_INTEGRATION not YES. Reopening an already-finalized stage uses existing
controlled stage revision/CAS protections. No new Registry lifecycle is defined.

## 8. Artifacts and cutoff discipline

Required artifacts are PRE_CERTIFICATION_QUESTION_LEDGER and
PRE_CERTIFICATION_CHALLENGE_REPORT, schema version 1.0 under this directory's
`schemas/`. Both identify run, cutoff, company, iteration, mode and exact lock
references. The report references the exact ledger ID/version/hash. Artifact
references include artifact_id, version and content_sha256. Candidate drops,
accepted-question judgments, origin triggers, materiality, evidence, effects,
mitigations, reopen targets, coverage and independent pass records remain durable.

The consuming validator must resolve exact artifact bytes and verify hash, run,
cutoff, active versions and dependency lineage; caller-supplied PASS is insufficient.
Any subsequent material upstream change invalidates admission. Hypothetical future
worlds are allowed and labelled hypothetical; actual evidence later than the
immutable DATA_CUTOFF is forbidden and requires governed refresh/successor routing.

## 9. Certification and Integration boundaries

Certification consumes the current passing report AND ledger, current locks and
underlying ledgers. Every CONCERN with its question ID, decision impact and
mitigation/limitation must propagate to Certification limitations and the final
thesis/invalidation triggers where relevant. PASS_WITH_CONCERNS is admission,
never a substitute Certification result or automatic score permission.

Certification keeps its existing evidence sufficiency, calculation reproducibility,
methodology compliance, traceability, formal cross-block reconciliation, dual
Certification decisions, score permission and terminal/readiness responsibilities.
Existing OQS, OVS, Investment Score, weights, caps, Elite thresholds, frozen Weak
Link treatment and terminal OroTitan conjunction remain unchanged. The proposed
deterministic scoring draft is excluded.

Only Certification may normally finalize Deep Dive. Integration is non-analytical:
verify final Deep Dive lineage, matching Challenge refs in Certification, current
PASS/PASS_WITH_CONCERNS, concern propagation and no intervening invalidation.
Missing, stale, superseded, REOPEN or FAIL artifacts block admission. Integration
must not generate questions, repair concerns or infer PASS from counts.

## 10. Historical firewall and activation

All runs preceding Method-V2 activation classify METHOD_V1, including historical
runtime V2/V3. Their original authority and snapshots remain immutable. This
freeze performs no database migration, runtime activation, run creation, snapshot
rewrite, publication or pointer mutation. Replay requires a newly authorized
controlled successor; the parent is never rebound. Separate implementation,
reviewed runtime bindings, clean-room validation and production authorization are
required before use in production. FROZEN does not mean PRODUCTION_ACTIVE.

## 11. Acceptance matrix and change control

Required rejection cases: absent Challenge; stale lock/ledger/report; wrong run,
cutoff or hash; unresolved REOPEN/FAIL; insufficient family/company coverage;
missing independent saturation evidence; new material questions remaining;
unpropagated concerns; post-cutoff evidence. Required positive cases: current
saturated PASS and PASS_WITH_CONCERNS with exact propagated limitations; completion
below the old draft quota; more questions when material yield continues.

Regression IDs M01–M22 in `tests/method-v2-authority.test.ts` cover composition,
hash firewall, historical classification, admission and no side effects. They are
offline authority tests, not proof of deployed Registry enforcement.

Changes to this frozen authority, membership, scopes or precedence require an
explicit successor version and project approval. Never alter historical files or
silently replace Method-V2 bytes. Future METHOD_V3 requires a distinct approved
manifest; runtime/schema version changes alone establish no Method generation.
