# OROTITAN ANALYTICAL ENGINE V2 — DATA CONTRACTS V0.1

Status: DESIGN CANDIDATE — NOT FROZEN  
Program: OroTitan Equity Research vNext  
Methodology change: NO  
Production mutation: NONE  
Supabase migration: NONE  
Primary analytical engine: ChatGPT under `OROTITAN_CHATGPT_OPERATING_PROTOCOL_V1_FREEZE_V1.0`

---

# 1. PURPOSE

This design translates already-frozen Research / Deep Dive / Integration semantics into machine-readable V2 analytical artifacts.

The objective is not to invent a second analytical methodology.

The objective is to make ChatGPT-produced research:

- exact;
- traceable;
- versionable;
- recoverable;
- cutoff-safe;
- conflict-aware;
- cross-referenceable;
- structurally validatable;
- suitable for controlled persistence;
- suitable for deterministic projection into the existing Phase-4 snapshot.

Core rule:

```text
FROZEN ANALYTICAL SEMANTICS
→ V2 MACHINE REPRESENTATION
→ CONTROLLED VALIDATION
→ PERSISTED ARTIFACTS
```

not:

```text
V2 SCHEMA
→ NEW ANALYTICAL SEMANTICS
```

---

# 2. AUTHORITY BOUNDARY

Authority remains:

1. Analysis Standard V1;
2. Research / Deep Dive / Integration frozen Stage Contracts;
3. frozen Research Execution Process;
4. run-specific contract pins;
5. persisted authoritative run artifacts.

These V2 schemas are an implementation layer.

If a schema cannot represent a frozen state without semantic loss:

```text
SCHEMA DEFECT
→ FIX SCHEMA
```

Never:

```text
SCHEMA DEFECT
→ CHANGE ANALYTICAL MEANING
```

---

# 3. PERSISTENCE BOUNDARY

The live registry already separates physical artifact identity from analytical payload.

Therefore V2 payload schemas do not replace:

- `orotitan_artifacts.artifact_id`;
- `version`;
- `content_sha256`;
- `authority_class`;
- `storage_uri`;
- manifest registration;
- artifact lineage edges.

Artifact registry = physical / authority identity.

Artifact payload = analytical content.

This prevents duplicate identity systems.

---

# 4. CONTRACT FAMILY

Initial V2 core contract family:

```text
OROTITAN_ANALYTICAL_COMMON
RESEARCH_SOURCE_MANIFEST
EVIDENCE_LEDGER
CONFLICT_LEDGER
CALCULATION_LEDGER
MATERIAL_ASSUMPTION_REGISTER
MATERIAL_RESEARCH_HYPOTHESIS_REGISTER
RESEARCH_GAP_REGISTER
DD_INPUT_SUFFICIENCY_RECORD
ANALYSIS_INPUT_LOCK
ANALYTICAL_BLOCK_OUTPUT
```

These are sufficient to close the first Data-Contract gaps in the protocol battery:

- post-cutoff evidence validation;
- V2 evidence / conflict structures;
- analytical Evidence-ID validation;
- exact cross-block traceability;
- Research → Deep Dive input lock.

V2-specific Company Economic DNA / Sector Overlay / Cyclicality / Technology / Causal Graph contracts are a second tranche built on this common core.

---

# 5. COMMON PAYLOAD CONTEXT

Every authoritative analytical payload carries a common context:

```text
schema_name
schema_version
run_id
stage_code
stage_revision
data_cutoff
contract_set_sha256
generated_at
```

Rules:

- `run_id` must equal the registry run.
- `stage_code` must equal the owning stage.
- `data_cutoff` must equal the run cutoff.
- `stage_revision` supports reopen history.
- `contract_set_sha256` binds the payload to the run's pinned method.
- `generated_at` is metadata and does not alter point-in-time evidence admissibility.

No payload-level field may silently override registry authority.

---

# 6. SPECIAL STATES

The existing canonical special-state vocabulary is preserved where a field genuinely accepts a special state:

```text
UNKNOWN
NOT_APPLICABLE
NOT_ASSESSABLE
MISSING
NOT_AVAILABLE
```

Do not collapse these to JSON null.

JSON null remains a transport-level absence only where explicitly permitted.

---

# 7. SOURCE MANIFEST

The Source Manifest is broader than the Evidence Ledger.

A discovered source may be registered without being promoted into material evidence.

Frozen minimum semantics preserved:

```text
SOURCE_ID
TITLE / DESCRIPTION
SOURCE_TYPE
ISSUER / PUBLISHER
SOURCE_DATE
DATA_PERIOD
ROOT_SOURCE_ID
URL / FILE_REFERENCE
DISCOVERY_ROUTE
ACCESS_STATUS
INCLUDED_IN_EVIDENCE_LEDGER
NOTES / LIMITATIONS
```

V0.1 deliberately leaves `source_type`, `discovery_route` and `access_status` as controlled non-empty strings rather than inventing new frozen enums.

Later implementation may introduce a versioned taxonomy without altering the Research Stage Contract.

---

# 8. EVIDENCE LEDGER

One authoritative Evidence Ledger exists per run lineage/version.

Exact frozen evidence vocabulary retained:

```text
EVIDENCE_ID
CLAIM_ID
CLAIM / METRIC
VALUE / STATEMENT
PERIOD / AS_OF_DATE
SOURCE
ROOT_SOURCE_ID
INDEPENDENCE_GROUP
SOURCE_CLASS
CLAIM_FIT
SOURCE_DATE
DATA_CUTOFF
EPISTEMIC_TYPE
FRESHNESS_STATE
LIMITATIONS
CONFLICT_STATUS
USED_IN[]
```

Exact frozen enums:

```text
SOURCE_CLASS
= S1 | S2 | S3 | S4 | S5 | S6

CLAIM_FIT
= HIGH | MEDIUM | LOW

EPISTEMIC_TYPE
= REPORTED
| CALCULATED
| CONSENSUS
| ESTIMATE
| ASSUMPTION
| UNKNOWN

FRESHNESS_STATE
= CURRENT
| FIT_FOR_PURPOSE
| STALE_FOR_USE
| UNKNOWN_DATE
```

## 8.1 Evidence value representation

V2 introduces a machine representation, not new semantics:

```text
EVIDENCE_VALUE
= TEXT
| NUMBER
| RANGE
| BOOLEAN
| SPECIAL_STATE
```

Numeric forms may additionally carry:

- unit;
- currency;
- accounting basis.

The evidence row retains separate `period` and `as_of_date`.

This exists to prevent silent comparisons across incompatible periods, units, currencies or accounting bases.

## 8.2 Point-in-time validator

JSON Schema validates structure.

A deterministic cross-artifact validator enforces:

```text
evidence.source_date <= run.data_cutoff
evidence.data_cutoff = run.data_cutoff
```

A post-cutoff source may exist in source discovery metadata but cannot be admitted as current-run evidence.

---

# 9. CONFLICT LEDGER

Exact frozen semantics preserved:

```text
CONFLICT_ID
METRIC / CLAIM
SOURCE_A
VALUE_A
EVIDENCE_ID_A
SOURCE_B
VALUE_B
EVIDENCE_ID_B
CONFLICT_TYPE
REASON
RESOLUTION
MATERIALITY
AFFECTED_OUTPUTS[]
```

Exact frozen enums:

```text
CONFLICT_TYPE
= DATE
| DEFINITION
| ACCOUNTING_BASIS
| UNIT
| RESTATEMENT
| FACTUAL
| OTHER

RESOLUTION
= RESOLVED | UNRESOLVED
```

`materiality` remains a non-empty semantic value in V0.1 because no higher-authority frozen enum was found.

Do not invent HIGH / MEDIUM / LOW as canonical conflict materiality merely for implementation convenience.

Cross-artifact validation requires both referenced Evidence IDs to resolve.

---

# 10. CALCULATION LEDGER

Exact frozen semantics preserved:

```text
CALCULATION_ID
METRIC_NAME
FORMULA
INPUTS[]
INPUT_EVIDENCE_IDS[]
UNITS
CURRENCY
PERIOD
BASIS
RAW_RESULT
ROUNDING_RULE
PUBLISHED_RESULT
METHOD_VERSION
CALCULATION_VERSION
AUTOMATED_CHECK_STATUS
LIMITATIONS
```

Material calculations must remain independently reproducible.

V0.1 does not create a new calculation engine.

Existing deterministic Gate-15 modules remain separate authorities where applicable.

---

# 11. MATERIAL ASSUMPTION REGISTER

Exact frozen semantics preserved:

```text
ASSUMPTION_ID
VARIABLE
VALUE / RANGE
EPISTEMIC_TYPE
SOURCE / RATIONALE
SENSITIVITY
USED_IN
```

An assumption cannot silently become reported evidence.

The V2 validator can reject any attempt to represent an assumption row with an epistemic type inconsistent with the frozen assumption boundary.

---

# 12. MATERIAL RESEARCH HYPOTHESIS REGISTER

Frozen minimum fields:

```text
HYPOTHESIS_ID
AFFECTED_ANALYTICAL_BLOCK
HYPOTHESIS
WHY_MATERIAL
SUPPORTING_EVIDENCE_IDS[]
CONTRADICTING_EVIDENCE_IDS[]
UNRESOLVED_POINTS[]
RESEARCH_STATUS
```

`research_status` remains a non-empty string in V0.1 because no frozen enum was found.

The contract must not convert a Research hypothesis into a Deep Dive verdict.

---

# 13. RESEARCH GAP REGISTER

Frozen minimum fields:

```text
GAP_ID
AFFECTED_INPUT_BLOCK
QUESTION
MATERIALITY
SEARCHES_ATTEMPTED[]
BEST_NEXT_SOURCE
STATUS
WHY_NOT_RESOLVED
IMPACT_ON_READY_FOR_DEEP_DIVE
```

`materiality` and `status` remain non-empty strings until a higher-authority vocabulary is explicitly pinned.

This register is essential to anti-loop behavior:

```text
SAME GAP
+ SAME SEARCH HISTORY
+ NO NEW SOURCE / METHOD
→ DO NOT REPEAT DEAD-END SEARCH
```

---

# 14. DD INPUT SUFFICIENCY RECORD

Exact allowed status:

```text
SUFFICIENT
INSUFFICIENT
NOT_APPLICABLE
```

Never:

```text
PARTIALLY_SUFFICIENT
```

Initial / Full Refresh required block IDs:

```text
IDENTITY_PIT_INPUTS
BUSINESS_MODEL_INPUTS
MOAT_INPUTS
RUNWAY_INPUTS
RETURN_QUALITY_INPUTS
FCF_FORENSIC_INPUTS
CAPITAL_ALLOCATION_INPUTS
MANAGEMENT_GOVERNANCE_INPUTS
OUTSIDE_VIEW_INPUTS
RISK_RESILIENCE_INPUTS
VALUATION_INPUTS
```

Each block preserves:

```text
BLOCK_ID
APPLICABILITY
DD_INPUT_STATUS
MANDATORY_COVERAGE
EVIDENCE_ADEQUACY
MATERIAL_BLOCKING_GAPS[]
KEY_EVIDENCE_IDS[]
KEY_CONFLICT_IDS[]
SEARCHES_PERFORMED[]
BEST_NEXT_SOURCE
RATIONALE
```

Cross-field validator rule:

```text
READY_FOR_DEEP_DIVE = YES
only if required scope is SUFFICIENT or validly NOT_APPLICABLE
and no critical blocker
and no material unresolved blocking conflict
and cutoff / version integrity passes
```

---

# 15. ANALYSIS INPUT LOCK

This is the exact Research → Deep Dive execution boundary.

Required conceptual fields preserved:

```text
RUN_ID
COMPANY_ID
DATA_CUTOFF
RESEARCH_STAGE_CONTRACT_VERSION
EVIDENCE_LEDGER_VERSION
SOURCE_MANIFEST_VERSION
DD_INPUT_SUFFICIENCY_VERSION
OPEN_NONCRITICAL_GAPS[]
OPEN_MATERIAL_CONFLICTS[]
CRITICAL_BLOCKERS[]
READY_FOR_DEEP_DIVE
CREATED_AT
```

V2 additionally binds exact artifact references for:

- source manifest;
- evidence ledger;
- conflict ledger;
- hypothesis register;
- gap register;
- DD sufficiency record.

These references do not change the frozen semantics; they remove filename guessing.

---

# 16. ANALYTICAL BLOCK OUTPUT

The frozen Deep Dive execution envelope is represented directly:

```text
BLOCK_ID
BLOCK_VERSION
EXECUTION_STATUS
EXECUTION_CONFIDENCE
CANONICAL_VERDICT_FIELDS[]
CORE_FINDINGS[]
SUPPORTING_EVIDENCE_IDS[]
CONTRADICTING_EVIDENCE_IDS[]
CALCULATION_IDS[]
MATERIAL_ASSUMPTION_IDS[]
CONFLICT_IDS[]
UNRESOLVED_POINTS[]
MATERIAL_DEPENDENCIES[]
REOPEN_TRIGGERS[]
RATIONALE
LAST_RESEARCH_DATE
DATA_CUTOFF
```

Exact execution enums:

```text
EXECUTION_STATUS
= INSUFFICIENT
| IN_PROGRESS
| PROVISIONALLY_STABLE
| LOCKED

EXECUTION_CONFIDENCE
= HIGH | MEDIUM | LOW
```

These are execution metadata only.

They must not replace Certification block states:

```text
COMPLETE
COMPLETE_WITH_LIMITATIONS
NOT_APPLICABLE
MISSING
```

## 16.1 Canonical verdict fields

To prevent an unrestricted narrative blob, V0.1 represents verdict fields as:

```text
field
value
evidence_ids[]
calculation_ids[]
assumption_ids[]
conflict_ids[]
```

The field name and value must still use the frozen analytical vocabulary of the owning block.

The schema does not define a new moat/runway/valuation vocabulary.

---

# 17. CROSS-ARTIFACT VALIDATOR

JSON Schema alone cannot enforce all OroTitan invariants.

A deterministic validator must later enforce at minimum:

## Identity

- payload run_id equals registry run_id;
- stage_code equals owning stage;
- stage_revision equals registry stage revision;
- contract_set_sha256 matches run pin set.

## Point-in-time

- source_date <= DATA_CUTOFF for admitted evidence;
- all payload cutoffs equal run cutoff.

## Referential integrity

- Evidence IDs resolve within authoritative Evidence Ledger;
- Conflict evidence references resolve;
- Calculation input Evidence IDs resolve;
- Hypothesis supporting / contradicting IDs resolve;
- DD sufficiency evidence / conflict IDs resolve;
- block output evidence / conflict / calculation / assumption IDs resolve.

## Source / Evidence consistency

- every Evidence row references a Source Manifest source;
- root_source_id resolves where supplied;
- `included_in_evidence_ledger = true` is consistent for used material sources.

## Sufficiency

- no unsupported READY_FOR_DEEP_DIVE = YES.

## Material-change revalidation

V2 contracts provide exact references required by the Process Engine to later prove that a changed material conclusion was revalidated.

The Process Engine owns the gate; the Data Contract supplies the traceability.

---

# 18. V2-SPECIFIC SECOND TRANCHE

After the core contracts validate, build contracts for:

```text
COMPANY_ECONOMIC_DNA
OVERLAY_SELECTION
SECTOR_PROFILE_REFERENCE
CYCLE_ANALYSIS
TECHNOLOGY_MAP
INDUSTRY_STRUCTURE
CAUSAL_GRAPH
```

These objects are analytical support structures.

They must feed canonical blocks rather than create parallel terminal semantics.

Examples:

```text
CYCLE_ANALYSIS
→ RETURN QUALITY / FCF / VALUATION

TECHNOLOGY_MAP
→ MOAT / RUNWAY / RISK

CAUSAL_GRAPH
→ TRACEABILITY ACROSS MATERIAL CLAIMS
```

---

# 19. FRENCH-FIRST PRODUCT RULE

Machine contracts remain English.

The OroTitan product surface maps them to French labels.

Example:

```text
INSUFFICIENT
→ "Insuffisant"

PROVISIONALLY_STABLE
→ "Provisoirement stable"

NOT_ASSESSABLE
→ "Non évaluable"
```

Do not localize canonical IDs inside persisted payloads.

---

# 20. DATABASE STRATEGY

V0.1 does not introduce one SQL table per analytical object.

Preferred physical strategy:

```text
IMMUTABLE ARTIFACT PAYLOAD
+ EXISTING ARTIFACT REGISTRY
+ ARTIFACT EDGES
+ GENERATED / EXTRACTED READ VIEWS WHERE QUERYING IS MATERIAL
```

This preserves the Physical Implementation Inventory rule:

```text
DO NOT DUPLICATE BUSINESS LOGIC INTO FLATTENED COLUMNS
```

No production migration is authorized by this document.

---

# 21. ACCEPTANCE TESTS

Before freezing these data contracts, validate at minimum:

1. valid Research rich-disclosure pack;
2. post-cutoff evidence rejected;
3. root-source chain preserved;
4. duplicate independence does not create false corroboration;
5. issuer / customer conflict resolves to Conflict Ledger;
6. unresolved material conflict prevents unsupported sufficiency;
7. assumption cannot masquerade as reported fact;
8. invalid Evidence ID fails;
9. calculation Evidence refs fail if unresolved;
10. Research hypothesis remains hypothesis;
11. Research gap can preserve unresolved state;
12. `PARTIALLY_SUFFICIENT` rejected;
13. all 11 initial DD input blocks required;
14. Analysis Input Lock cannot claim ready with inconsistent sufficiency;
15. analytical block preserves UNKNOWN / NOT_ASSESSABLE;
16. block references exact Evidence / Calculation / Assumption / Conflict IDs;
17. Company Economic DNA later activates overlays without becoming a score;
18. serial-acquirer data structures can represent acquisition capital / cohort economics;
19. cycle structures distinguish reported vs normalized economics;
20. technology structures can represent substitution / replication without inferring moat;
21. canonical snapshot projection remains downstream and unchanged;
22. French-first UI can translate labels without mutating stored enums.

---

# 22. DESIGN STATUS

```text
DOCUMENT = OROTITAN_ANALYTICAL_ENGINE_V2_DATA_CONTRACTS_V0.1
STATUS = DESIGN_CANDIDATE
METHODOLOGY_CHANGE = NO
PRODUCTION_MUTATION = NO
SUPABASE_MIGRATION = NO
CURRENT_TRANCHE = CORE_ANALYTICAL_ARTIFACTS
NEXT = IMPLEMENT_CORE_SCHEMAS_AND_CROSS_ARTIFACT_VALIDATOR
```
