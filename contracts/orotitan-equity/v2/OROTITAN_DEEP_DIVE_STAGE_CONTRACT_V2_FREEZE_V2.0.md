# OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2 — FREEZE V2.0

**Status:** FROZEN DESIGN — V2.0  
**Depends on:** V2 Process + V2 Pilotage + V2 Research  
**Registry stage code:** `DEEP_DIVE`  
**Methodology change:** NO  
**Scoring / valuation formula change:** NO

## 0. Purpose

V2 Deep Dive converts an admitted evidence base into a fully traceable OroTitan analytical dossier while separating three cognitive tasks:

```text
PHASE 1 = FUNDAMENTALS
PHASE 2 = VALUATION
PHASE 3 = CERTIFICATION_RECONCILIATION
```

The phases may occupy separate discussions, but they remain one persistent `DEEP_DIVE` Registry stage.

Only Phase 3 may complete the stage and declare `READY_FOR_INTEGRATION = YES`.

## 1. Admission

Deep Dive may start only if authoritative persisted Research state shows:

```text
RESEARCH_STAGE_STATUS = COMPLETE
READY_FOR_DEEP_DIVE = YES
Research active manifest = FINAL
required Research artifacts = exact, available, hash-verified
no blocking Research condition
```

Failure -> stop -> return to Pilotage.

## 2. Shared Deep Dive rules

All phases obey:

```text
same RUN_ID
same DATA_CUTOFF
same immutable contract pins
same authoritative Evidence Ledger lineage
same frozen Evidence / Conflict / Calculation / Assumption semantics
```

New material evidence discovered during Deep Dive may be added only through the authoritative evidence lineage with normal source normalization and cutoff discipline.

Every changed analytical artifact body creates a new immutable version. No silent overwrite.

## 3. Phase 1 — Fundamentals

### 3.1 Question

> What business are we actually analyzing, what are its economic characteristics, how strong and durable are they, and what can break the fundamental case before price is considered?

### 3.2 Required blocks

Execute, with applicable sector overlays:

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

### 3.3 Price firewall

Fundamentals MUST NOT perform:

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

Market price may exist as run metadata but must not determine a fundamental judgment.

### 3.4 Scoring firewall

Fundamentals may produce the frozen methodology’s underlying dimension judgments/scoring inputs, but final:

```text
OQS_RAW
OQS
QUALITY_CLASS
```

must remain uncomputed/unpublished until Certification.

### 3.5 Fundamental Red Team

Red Team must challenge at minimum:

```text
moat mechanism and durability
runway assumptions
return attribution
cash quality / SBC / accounting normalization
capital allocation
management / governance
risk transmission mechanisms
outside-view contradictions
```

A material unresolved contradiction blocks `FUNDAMENTALS_LOCK`.

### 3.6 Taxonomy and business description

Fundamentals validates and locks the controlled taxonomy projection and produces `BUSINESS_DESCRIPTION_SHORT` under the V2 Process rules.

### 3.7 FUNDAMENTALS_LOCK

Required conceptual contents:

```text
exact Research input artifact refs
updated Evidence / Conflict / Calculation / Assumption refs where applicable
controlled taxonomy projection
BUSINESS_DESCRIPTION_SHORT
fundamental block outputs
Red Team record
dimension scoring inputs
material limitations
fundamental invalidation triggers
internal gate READY_FOR_VALUATION
```

On success:

```text
FUNDAMENTALS_LOCK = SEALED / AVAILABLE
READY_FOR_VALUATION = YES
DEEP_DIVE lifecycle = IN_PROGRESS
active manifest kind = CHECKPOINT
READY_FOR_INTEGRATION != YES
```

The final visible response block is the exact Valuation bootstrap.

## 4. Phase 2 — Valuation

### 4.1 Admission

Valuation requires exact resolution of:

```text
current FUNDAMENTALS_LOCK version
READY_FOR_VALUATION = YES
full authoritative Evidence Ledger lineage
Conflict Ledger
Calculation Ledger
Material Assumption Register
run / security / reference-price identity
frozen valuation policy / conventions
```

### 4.2 Question

> Given the locked fundamental case and evidence, what is the business worth and what return does the current reference price imply under the frozen valuation method?

### 4.3 Required outputs

Valuation owns, as applicable:

```text
economic discount rate
intrinsic-value scenarios
expected-return model
Mature Normalization
Same-Multiple diagnostic only where allowed
Reverse DCF
market-expectation gap
margin of safety
valuation reliability
price ladder
exact valuation inputs required by deterministic I2
```

### 4.4 Upstream contradiction rule

Valuation may not silently alter Fundamentals.

If material contradiction is found:

```text
identify exact issue + scope
persist reopen reason
reopen affected Fundamentals scope only
preserve prior lock/history
issue new FUNDAMENTALS_LOCK version
invalidate/supersede dependent valuation eligibility
rerun Valuation against new exact lock
```

Ambiguous scope -> return to Pilotage.

### 4.5 Score visibility

Valuation may compute reproducible provisional values necessary for control, but certified final `OVS` and `INVESTMENT_SCORE` are withheld from user-facing analytical conclusion until Certification.

### 4.6 VALUATION_LOCK

Required conceptual contents:

```text
exact FUNDAMENTALS_LOCK ref/version/hash
exact evidence / calculation / assumption lineage
valuation basis and methods
all material valuation calculations at full persisted precision
expected-return outputs
normalization selection inputs
reverse DCF
MOS / reliability
price ladder
I2 deterministic scoring inputs
material valuation limitations
internal gate READY_FOR_CERTIFICATION
```

On success:

```text
VALUATION_LOCK = SEALED / AVAILABLE
READY_FOR_CERTIFICATION = YES
DEEP_DIVE lifecycle = IN_PROGRESS
active manifest kind = CHECKPOINT
READY_FOR_INTEGRATION != YES
```

The final visible response block is the exact Certification bootstrap.

## 5. Phase 3 — Certification / Reconciliation

### 5.1 Admission

Certification requires:

```text
exact FUNDAMENTALS_LOCK = valid
exact VALUATION_LOCK = valid
READY_FOR_CERTIFICATION = YES
all required ledgers and artifacts = hash-resolvable
```

### 5.2 Non-creative rule

Certification is control, not a third analytical rewrite.

It MUST NOT:

```text
perform broad new research
invent missing evidence
silently modify a locked fundamental judgment
silently modify valuation assumptions
choose numbers merely to make scores reconcile
introduce an override to improve a result
```

Material missing/contradictory input -> reopen exact upstream scope.

### 5.3 Responsibilities

Certification owns:

```text
CROSS_BLOCK_RECONCILIATION
BUSINESS_RESEARCH_STATUS
INVESTMENT_CONCLUSION_STATUS
SCORE_PERMISSION
FORENSIC_RELIABILITY reconciliation
VALUATION_RELIABILITY reconciliation
OQS / QUALITY_CLASS
OVS deterministic reconciliation
INVESTMENT_RAW / INVESTMENT_SCORE / class
OROTITAN terminal conjunctive gate
DOSSIER_READINESS
OPPORTUNITY_PATH / PRICE_LADDER_STATUS where applicable
NEXT_ACTION
structured investment thesis
final invalidation triggers
final Deep Dive report
```

### 5.4 OQS timing

Final OQS may be calculated/displayed only after required certification state and score permission are established.

This is a presentation/execution sequencing change only. Frozen formulas, caps, thresholds and weak-link semantics are unchanged.

### 5.5 Structured investment thesis

Produce exactly:

```text
QUALITY_CASE
VALUATION_CASE
KEY_RISK
```

Each should normally be <= 240 characters and traceable to certified outputs.

### 5.6 Deterministic scoring precision

Persist scoring inputs at the precision actually used. Rounded presentation values must never become hidden higher-precision calculation inputs.

If deterministic I2 outputs do not exactly reconcile under the pinned implementation, Certification/Integration must fail closed and route to the appropriate upstream artifact version rather than silently correcting values.

## 6. Required V2 Deep Dive artifacts

At final completion, authoritative Deep Dive lineage must conceptually include:

```text
1. FUNDAMENTALS_LOCK
2. FUNDAMENTALS_ANALYTICAL_BLOCK_OUTPUTS
3. FUNDAMENTAL_RED_TEAM_RECORD
4. CONTROLLED_TAXONOMY_RECORD
5. BUSINESS_DESCRIPTION_SHORT_ARTIFACT
6. UPDATED EVIDENCE_LEDGER VERSION / REFERENCES
7. CONFLICT_LEDGER VERSION / REFERENCES
8. CALCULATION_LEDGER
9. MATERIAL_ASSUMPTION_REGISTER
10. VALUATION_LOCK
11. VALUATION_ARTIFACT
12. CROSS_BLOCK_RECONCILIATION_RECORD
13. CERTIFICATION_ARTIFACT
14. SCORING_INPUTS / OUTPUTS when permitted
15. OROTITAN_TERMINAL_GATE_ARTIFACT
16. READINESS_NEXT_ACTION_ARTIFACT
17. STRUCTURED_INVESTMENT_THESIS_ARTIFACT
18. DEEP_DIVE_REPORT
19. DEEP_DIVE_STAGE_MANIFEST
```

Physical layout is implementation detail. No duplicate ledger semantics are introduced.

## 7. Checkpoint protocol

Fundamentals and Valuation handoffs are durable Deep Dive checkpoints:

```text
persist sealed phase outputs
build CHECKPOINT Deep Dive Stage Manifest
hash-verify stored bytes
register artifacts + manifest
set active_manifest_kind = CHECKPOINT
preserve lifecycle IN_PROGRESS
handoff gate remains NOT_EVALUATED or NO
```

A CHECKPOINT never admits Integration.

## 8. Finalization protocol

Only Certification may perform normal Deep Dive finalization:

```text
required work complete
self-audit complete
final artifact bodies generated
bytes persisted / versioned
hash verification complete
Registry reconciled
FINAL Deep Dive Stage Manifest registered
DEEP_DIVE lifecycle = COMPLETE
READY_FOR_INTEGRATION = YES
```

Narrative answer alone never completes Deep Dive.

## 9. Reopening

Controlled reopening preserves history:

```text
stage_revision += 1 as required by Registry semantics
handoff gate -> NOT_EVALUATED
old FINAL / CHECKPOINT artifacts remain resolvable
reason and affected scope persisted
dependent eligibility invalidated
new versions created
```

No unrelated block may be reopened automatically.

## 10. Self-audit before final completion

Certification must verify at minimum:

```text
correct run / issuer / security / dossier
DATA_CUTOFF respected
exact Research artifacts loaded
Fundamentals Lock exact and current
Fundamental Red Team complete
Valuation Lock exact and current
new evidence normalized into authoritative lineage
material conflicts registered
facts and assumptions separated
calculations reproducible at persisted precision
sector overlays applied correctly
cross-block reconciliation complete
valuation basis / normalization selection exact
Certification before score publication
SCORE_PERMISSION obeyed
I2 deterministic outputs reconciled
terminal gate conjunctive
readiness separate from investability
all material conclusions traceable
no summary-as-authority
no post-cutoff contamination
```

Any material failure blocks normal completion unless represented through an explicitly allowed frozen limitation/state.

## 11. Final handoff

Success -> final visible block is exact Integration bootstrap, with no prose after it.

Blocked -> final visible block is exact resolution/Pilotage bootstrap, with no Integration prompt.
