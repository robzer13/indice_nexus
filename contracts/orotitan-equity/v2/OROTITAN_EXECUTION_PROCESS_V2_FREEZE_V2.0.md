# OROTITAN_EXECUTION_PROCESS_V2 — FREEZE V2.0

**Project:** OroTitan Equity Research  
**Status:** FROZEN DESIGN — V2.0  
**Freeze date:** 2026-09-15  
**Methodology change:** NO  
**Scoring formula change:** NO  
**Valuation convention change:** NO  
**I2 / I3-B change:** NO  
**V1 history rewrite:** FORBIDDEN  
**Production activation:** NOT AUTHORIZED BY THIS DOCUMENT ALONE

## 0. Purpose

V2 improves execution quality, separation of concerns and mobile usability without changing the frozen analytical methodology.

The V2 execution sequence is:

```text
PILOTAGE
→ RESEARCH
→ DEEP DIVE / FUNDAMENTALS
→ DEEP DIVE / VALUATION
→ DEEP DIVE / CERTIFICATION_RECONCILIATION
→ INTEGRATION
→ GO PUBLISH
```

The persistent Registry stage codes remain unchanged:

```text
RESEARCH
DEEP_DIVE
INTEGRATION
```

The three Deep Dive phases are authoritative internal phases of the existing `DEEP_DIVE` stage. They use immutable checkpoint artifacts and CHECKPOINT Stage Manifests. Only Certification/Reconciliation may finalize `DEEP_DIVE` and emit the FINAL Deep Dive Stage Manifest.

## 1. Core invariants

```text
ONE RUN
ONE DATA_CUTOFF
ONE CONTRACT SET
ONE AUTHORITATIVE EVIDENCE LINEAGE
ONE DEEP_DIVE REGISTRY STAGE
THREE DEEP_DIVE EXECUTION PHASES
FIVE EXECUTION DISCUSSIONS MAXIMUM
ONE CANONICAL SNAPSHOT AFTER AUTHORIZED PUBLICATION
```

The epistemic chain remains:

> Evidence → Calculation → Judgment → Narrative

V2 must never become score-first, valuation-first, thesis-confirmation, summary-as-authority or chat-memory reconstruction.

## 2. Discussion model

Pilotage recommends at most:

```text
<COMPANY> — RESEARCH — <INITIAL|REFRESH> <YYYY-MM>
<COMPANY> — FUNDAMENTALS — <INITIAL|REFRESH> <YYYY-MM>
<COMPANY> — VALUATION — <INITIAL|REFRESH> <YYYY-MM>
<COMPANY> — CERTIFICATION — <INITIAL|REFRESH> <YYYY-MM>
<COMPANY> — INTEGRATION — <INITIAL|REFRESH> <YYYY-MM>
```

`00 — PILOTAGE` remains permanent orchestration and never analytical truth. `RUN_ID` remains the machine identity.

## 3. Mandatory handoff UX

Every successful execution discussion MUST end with exactly one copy-ready prompt for the next discussion.

```text
SUCCESS
→ exact next-discussion prompt
→ prompt is final visible block
→ no prose after prompt

BLOCKED
→ exact resolution / return-to-Pilotage prompt
→ prompt is final visible block
→ no downstream prompt
```

Pilotage must be able to reconstruct the same prompt automatically from Registry and exact artifact state. Both workflows are therefore valid:

```text
A. USER COPY / PASTE
B. PILOTAGE AUTO-RECONSTRUCTION
```

Minimum bootstrap fields:

```text
COMPANY
RUN_ID
CANONICAL_MODE
RUN_TYPE
REGISTRY_STAGE
EXECUTION_PHASE
DATA_CUTOFF
EXPECTED_STAGE_CONTRACT
EXPECTED_STAGE_CONTRACT_VERSION
EXPECTED_INPUT_ARTIFACT_IDS / VERSIONS
BASELINE_SNAPSHOT_ID if applicable
HIGHER_AUTHORITY_PROCESS_VERSION
```

Mandatory bootstrap invariant:

```text
DO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE.
LOAD AND VERIFY THE AUTHORITATIVE CONTRACT AND ARTIFACTS BEFORE WORK.
FAIL CLOSED ON VERSION / ID / STATUS / HASH MISMATCH.
```

## 4. Research

Research remains evidence acquisition, normalization, source diversification, conflict registration, gap closure and input sufficiency.

Research MUST NOT produce final moat/runway/business-quality verdicts, valuation, OQS, OVS, Investment Score, terminal OroTitan status or investment thesis.

Research gathers the factual classification inputs required by the V2 taxonomy. Fundamentals locks the final classification projection used downstream.

Normal admission to Fundamentals requires:

```text
RESEARCH_STAGE_STATUS = COMPLETE
FINAL Research Stage Manifest = VALID
READY_FOR_DEEP_DIVE = YES
required Research artifacts = exact and hash-resolvable
```

## 5. Deep Dive Phase 1 — Fundamentals

Central question:

> What business are we analyzing, how good is it economically, why, and what can invalidate that conclusion before price is considered?

Fundamentals owns:

```text
BUSINESS MODEL
ECONOMIC QUALITY
MOAT
RUNWAY
RETURN QUALITY
FCF / OWNER EARNINGS / FORENSIC
CAPITAL ALLOCATION
MANAGEMENT / GOVERNANCE
OUTSIDE VIEW
RISK / RESILIENCE
FUNDAMENTAL RED TEAM / PRE-MORTEM
```

The Fundamental Red Team must be completed before the phase can lock.

Fundamentals MUST NOT perform or conclude:

```text
DCF
FAIR VALUE
EXPECTED RETURN FROM CURRENT PRICE
REVERSE DCF PRICE CONCLUSION
PRICE LADDER
OVS
INVESTMENT SCORE
OROTITAN TERMINAL GATE
INVESTMENT THESIS
```

Fundamentals may produce the analytical dimension inputs required later for deterministic scoring, but final `OQS_RAW`, `OQS` and `QUALITY_CLASS` MUST NOT be calculated or displayed before Certification.

Normal completion creates immutable `FUNDAMENTALS_LOCK` containing exact evidence/calculation/assumption references, business-model conclusion, taxonomy, short business description, fundamental block conclusions, Red Team, dimension scoring inputs, limitations and invalidation triggers.

Internal gate:

```text
READY_FOR_VALUATION = YES | NO
```

`FUNDAMENTALS_LOCK` is a Deep Dive checkpoint artifact. The associated Stage Manifest is `CHECKPOINT`; Registry `DEEP_DIVE` remains `IN_PROGRESS` and `READY_FOR_INTEGRATION` remains `NOT_EVALUATED` or `NO`.

### Business description

`BUSINESS_DESCRIPTION_SHORT` rules:

```text
1–3 sentences
maximum 450 characters
factual and neutral
what is sold, to whom, how money is made
no valuation language
no score language
no investment recommendation
```

## 6. Deep Dive Phase 2 — Valuation

Central question:

> Given the locked fundamental case and authoritative evidence, what is the business worth and what return does the reference price imply under the frozen valuation policy?

Mandatory inputs include BOTH:

```text
exact FUNDAMENTALS_LOCK version
full authoritative Evidence Ledger lineage
Conflict Ledger
Calculation Ledger
Material Assumption Register
run identity / DATA_CUTOFF / reference-price inputs
locked investment policy / valuation conventions
```

Valuation may challenge a locked fundamental input but MUST NOT silently rewrite it.

Material contradiction:

```text
FUNDAMENTAL_CONTRADICTION = MATERIAL
→ limited automatic Fundamentals reopen
→ persist exact reason and affected scope
→ new FUNDAMENTALS_LOCK version
→ dependent Valuation outputs superseded / invalidated
→ rerun Valuation against new exact lock
```

Valuation owns economic discount rate, intrinsic value, expected returns, mature normalization, permitted same-multiple diagnostic, reverse DCF, market expectation gap, margin of safety, valuation reliability, price ladder and valuation inputs required by I2.

Final certified OVS and Investment Score remain withheld until Certification.

Normal completion creates immutable `VALUATION_LOCK` plus CHECKPOINT Deep Dive Stage Manifest.

Internal gate:

```text
READY_FOR_CERTIFICATION = YES | NO
```

## 7. Deep Dive Phase 3 — Certification / Reconciliation

Central question:

> Are the locked Fundamentals and Valuation conclusions internally consistent, traceable and certifiable, and what deterministic scores, terminal state and next action follow?

Certification is a non-creative control phase. It MUST NOT perform broad new research, invent missing facts, improve a score by changing an upstream judgment or silently rewrite an upstream lock.

If new material evidence is required, route to the relevant upstream phase.

Certification owns:

```text
cross-block reconciliation
business research certification
investment conclusion certification
score permission
forensic / valuation reliability reconciliation
OQS computation
OVS deterministic reconciliation
Investment Score computation
OroTitan terminal gate
Dossier Readiness
Next Action
structured investment thesis
final invalidation triggers
final Deep Dive report
```

Final OQS, OVS and Investment Score become visible only here, after certification state and score permission are valid.

### Structured investment thesis

Certification produces exactly:

```text
QUALITY_CASE
VALUATION_CASE
KEY_RISK
```

Each field should normally fit within 240 characters.

### Final Deep Dive admission gate

Only Certification may emit the FINAL Deep Dive Stage Manifest.

Deep Dive becomes `COMPLETE` only after:

```text
valid FUNDAMENTALS_LOCK
valid VALUATION_LOCK
Certification complete
required Deep Dive artifacts persisted / versioned
Registry reconciled
FINAL Deep Dive Stage Manifest registered
READY_FOR_INTEGRATION = YES
```

Only that state admits Integration.

## 8. Deep Dive checkpoint lineage

```text
Fundamentals complete
→ FUNDAMENTALS_LOCK vN
→ CHECKPOINT Deep Dive Manifest A
→ DEEP_DIVE remains IN_PROGRESS
→ READY_FOR_INTEGRATION not admitted

Valuation complete
→ VALUATION_LOCK vN
→ CHECKPOINT Deep Dive Manifest B
→ DEEP_DIVE remains IN_PROGRESS
→ READY_FOR_INTEGRATION not admitted

Certification complete
→ final Deep Dive artifact set
→ FINAL Deep Dive Manifest C
→ DEEP_DIVE = COMPLETE
→ READY_FOR_INTEGRATION = YES
```

CHECKPOINT never admits Integration. Historical checkpoint artifacts remain hash-resolvable for audit.

## 9. Controlled taxonomy V2

Taxonomy exists for classification, filtering and portfolio navigation only. It has zero scoring authority.

Canonical classification uses:

```text
issuer_country_code = ISO 3166-1 alpha-2
primary_listing_country_code = ISO 3166-1 alpha-2
sector = controlled list
industry_group = controlled list
business_model_primary = controlled list
business_model_secondary = controlled optional value
economic_exposure_regions = controlled multi-select
taxonomy_version = required
```

No free-text country, sector, industry group or business-model value is allowed in canonical V2 projection. Unknown industry classification uses controlled `OTHER`, not arbitrary text.

Economic exposure is separate from legal domicile. Percentages MUST NOT be invented when not disclosed.

Optional portfolio filter:

```text
PEA_ELIGIBILITY = YES | NO | UNKNOWN
```

If assessed, it must carry an as-of date and source reference. PEA eligibility has zero analytical or scoring authority.

The exact lists are frozen in `OROTITAN_TAXONOMY_V2.0.json`.

## 10. Refresh V2

The existing frozen classifier is preserved:

```text
PRICE_ONLY_DELTA
ROUTINE_FUNDAMENTAL_DELTA
FULL_REFRESH_REQUIRED
```

### PRICE_ONLY_DELTA

Research may perform a minimal delta/revalidation pass. If no material fundamental change is found, the prior certified Fundamentals Lock may be explicitly `REVALIDATED` for the successor run. Fundamentals discussion may be skipped; Valuation, Certification and Integration execute as required. OQS cannot change without an authorized fundamental reopen.

### ROUTINE_FUNDAMENTAL_DELTA

Reopen only affected fundamental blocks and material downstream dependencies, then rerun materially affected Valuation, Certification and Integration.

### FULL_REFRESH_REQUIRED

Run the full V2 chain.

Refresh remains targeted. Analyst impact assessment remains authoritative within the frozen classifier.

## 11. V1 grandfathering

Existing V1 canonical snapshots and runs remain immutable historical truth.

```text
QUALYS CURRENT V1 SNAPSHOT
→ remains canonical until a future authorized refresh

NEXT QUALYS REFRESH
→ successor run under V2 pins
→ new immutable snapshot if publication succeeds
```

No retroactive migration exists merely to make V1 dossiers look V2-shaped.

## 12. Integration V2

Integration remains non-analytical. It consumes the exact FINAL Deep Dive artifact set and MUST NOT reconstruct missing analysis.

Responsibilities remain:

```text
canonical mapping
schema validation
I2 deterministic reconciliation
history transition validation
I3-B admission
canonical snapshot candidate
pre-publication control card
READY_TO_PUBLISH
```

Integration projects V2 taxonomy, short business description and structured investment thesis into the canonical snapshot.

No production promotion occurs without separate:

```text
GO PUBLISH <COMPANY>
```

## 13. Canonical product additions

At minimum V2 canonical product truth must expose:

```text
CLASSIFICATION
- issuer_country_code
- primary_listing_country_code
- sector
- industry_group
- business_model_primary
- business_model_secondary
- economic_exposure_regions
- taxonomy_version

BUSINESS SUMMARY
- business_description_short

INVESTMENT THESIS
- quality_case
- valuation_case
- key_risk

OPTIONAL PORTFOLIO FILTER
- pea_eligibility
- pea_eligibility_as_of
```

These fields are descriptive/operational outputs and do not alter OQS, OVS, Investment Score or terminal OroTitan logic.

## 14. Auto-limited reopening

A downstream phase may automatically reopen upstream work only if all are true:

```text
material contradiction identified
exact affected scope identified
reason persisted
prior artifacts preserved
new artifact versions created
all dependent downstream outputs invalidated or superseded
no unrelated block reopened
```

If scope is ambiguous, return to Pilotage instead of broadening automatically.

## 15. Freeze governance

V2.0 exists specifically to stop company-by-company process drift.

After V2 production activation:

```text
NO ROUTINE PROCESS CHANGE
NO COMPANY-SPECIFIC EXCEPTION
NO TAXONOMY SEMANTIC CHANGE
NO HANDOFF-SEQUENCE CHANGE
NO SCORE / VALUATION / CERTIFICATION CHANGE
```

Observations go to a backlog.

Normal V2 review opens only after BOTH:

```text
at least 10 completed V2 company analyses
AND
at least 6 weeks since first V2 production run
```

A V2.0.x emergency patch before that gate is allowed only for a genuinely blocking security, data-integrity, persistence/Registry, contract-contradiction or deterministic canonical-output defect.

Not emergency reasons: one awkward company, layout preference, a new analytical idea, changing weights, thresholds, valuation conventions or adding discretionary overrides.

## 16. Minimum acceptance matrix

V2 is not production-executable until implementation proves at minimum:

```text
V2-01 Research cannot emit OQS / OVS / Investment Score.
V2-02 Research FINAL + READY_FOR_DEEP_DIVE required before Fundamentals.
V2-03 Fundamentals cannot emit DCF / expected return / price-ladder conclusion.
V2-04 Fundamental Red Team complete before FUNDAMENTALS_LOCK.
V2-05 FUNDAMENTALS_LOCK immutable and hash-resolvable.
V2-06 Final OQS absent before Certification.
V2-07 Valuation consumes exact FUNDAMENTALS_LOCK version.
V2-08 Valuation also consumes authoritative Evidence Ledger lineage.
V2-09 Valuation cannot silently rewrite Fundamentals.
V2-10 Material contradiction triggers limited Fundamentals reopen.
V2-11 New Fundamentals Lock supersedes dependent Valuation eligibility.
V2-12 Valuation CHECKPOINT cannot admit Integration.
V2-13 Certification cannot perform broad new research.
V2-14 Certification routes missing material evidence upstream.
V2-15 Certification computes OQS deterministically from certified inputs.
V2-16 Certification reconciles OVS / Investment Score exactly under I2.
V2-17 Terminal gate only after valid certification state.
V2-18 FINAL Deep Dive Manifest only from Certification.
V2-19 CHECKPOINT Deep Dive Manifest cannot admit Integration.
V2-20 FINAL + COMPLETE + READY_FOR_INTEGRATION=YES admits Integration.
V2-21 Successful discussion ends with exactly one copy-ready next prompt.
V2-22 No visible prose follows handoff prompt.
V2-23 Blocked discussion emits resolution prompt, not downstream prompt.
V2-24 Pilotage reconstructs same prompt from Registry state.
V2-25 Country taxonomy rejects non-ISO free text.
V2-26 Sector taxonomy rejects non-controlled value.
V2-27 Business model taxonomy rejects non-controlled value.
V2-28 BUSINESS_DESCRIPTION_SHORT obeys length and neutrality rules.
V2-29 Thesis contains exactly QUALITY_CASE / VALUATION_CASE / KEY_RISK.
V2-30 Thesis is not produced before Certification.
V2-31 PRICE_ONLY_DELTA cannot change OQS without fundamental reopen.
V2-32 Targeted refresh yields new immutable snapshot after reconciliation.
V2-33 V1 Qualys remains unchanged until later refresh.
V2-34 V1 historical runs remain contract-resolvable after V2 activation.
V2-35 Registry stage-code constraints remain satisfied.
V2-36 Deep Dive internal checkpoints remain DEEP_DIVE artifacts/manifests.
V2-37 No new Registry table required solely for phase splitting.
V2-38 Integration projects taxonomy/description/thesis without analytical rewrite.
V2-39 I2 / I3-B remain fail-closed.
V2-40 No publication without GO PUBLISH.
V2-41 Contract pin mismatch fails closed.
V2-42 DATA_CUTOFF immutable within run.
V2-43 Evidence / Conflict / Calculation / Assumption semantics remain single-authority.
V2-44 Company-specific process exception rejected during freeze window.
V2-45 Emergency patch requires documented blocking-defect classification.
```

## 17. Compatibility result

```text
ANALYTICAL METHOD             UNCHANGED
SCORING FORMULAS              UNCHANGED
VALUATION POLICY              UNCHANGED
I2                            UNCHANGED
I3-B                          UNCHANGED
EVIDENCE AUTHORITY            SINGLE LINEAGE
REGISTRY TABLE MODEL          UNCHANGED
REGISTRY STAGE CODES          UNCHANGED
CHECKPOINT SEMANTICS          REUSED
DEEP_DIVE FINAL GATE          UNCHANGED
GO PUBLISH BOUNDARY           UNCHANGED
V1 HISTORY                    PRESERVED
```

Material process changes are limited to Deep Dive sequencing, score visibility timing, mandatory deterministic handoff UX, controlled taxonomy/product descriptors and freeze governance.

## 18. Mobile-first target flow

```text
GO <COMPANY>
→ Research
→ exact FUNDAMENTALS prompt
→ Fundamentals + Red Team + FUNDAMENTALS_LOCK
→ exact VALUATION prompt
→ Valuation + VALUATION_LOCK
→ exact CERTIFICATION prompt
→ Certification / reconciliation / scores / terminal gate / thesis / FINAL Deep Dive Manifest
→ exact INTEGRATION prompt
→ Integration / I2 / I3-B / READY_TO_PUBLISH
→ exact GO PUBLISH <COMPANY>
→ authorized canonical promotion
→ Vercel reads new canonical snapshot automatically
```

The user never manually reconstructs analytical context.

## 19. Status

```text
DOCUMENT = OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0
STATUS = FROZEN DESIGN
FREEZE_DATE = 2026-09-15
METHODOLOGY_CHANGE = NO
V1_REWRITE = NO
REGISTRY_DDL_REQUIRED_FOR_PHASE_SPLIT = NO
PRODUCTION_ACTIVATION = BLOCKED UNTIL V2 IMPLEMENTATION ADMISSION PASSES
```
