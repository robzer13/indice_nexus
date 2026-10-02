# OROTITAN ANALYTICAL ENGINE V2 — DATA CONTRACTS DESIGN V0.1

Status: DETAILED DESIGN CANDIDATE — NOT FROZEN  
Date: 2026-10-01  
Parent architecture:
- `OROTITAN_ANALYTICAL_ENGINE_V2_DESIGN_V0.1`
- `OROTITAN_CHATGPT_OPERATING_PROTOCOL_V1_FREEZE_V1.0`

Production mutation authority: NONE

---

# 1. PURPOSE

Define the machine-readable analytical contracts required to make the frozen ChatGPT operating protocol:

- resumable;
- traceable;
- auditable;
- anti-loop;
- cutoff-safe;
- compatible with existing stage artifacts;
- suitable for a French-first OroTitan product surface.

This design does **not** create a new methodology, score or stage authority.

The frozen Research / Deep Dive / Integration artifacts remain authoritative.

The new V2 contracts structure the analytical content that feeds those artifacts.

---

# 2. NON-NEGOTIABLE BOUNDARY

```text
V2 ANALYTICAL OBJECTS
→ enrich / structure analytical work
→ project into frozen stage artifacts

V2 ANALYTICAL OBJECTS
≠ replacement stage authority
```

Examples:

```text
EvidenceItem[]
→ EVIDENCE_LEDGER

Conflict[]
→ CONFLICT_LEDGER

OpenQuestion[]
→ RESEARCH_GAP_REGISTER

Calculation[]
→ CALCULATION_LEDGER

Assumption[]
→ MATERIAL_ASSUMPTION_REGISTER

AnalyticalBlockOutput[]
→ ANALYTICAL_BLOCK_OUTPUTS
```

Integration remains deterministic.

---

# 3. CONTRACT LAYERS

The data model is divided into six layers.

## L0 — RUN IDENTITY / LOCK

Inherited from the Registry:

- RUN_ID;
- STAGE;
- STAGE_REVISION;
- ISSUER_ID;
- SECURITY_ID;
- DOSSIER_ID;
- RUN_TYPE;
- CANONICAL_MODE;
- DATA_CUTOFF;
- CONTRACT_SET_SHA256;
- CONTRACT_PINS.

These values are not authored by ChatGPT.

## L1 — RESEARCH PRIMITIVES

- SOURCE_RECORD;
- EVIDENCE_ITEM;
- CONFLICT_RECORD;
- RESEARCH_GAP / OPEN_QUESTION;
- MATERIAL_RESEARCH_HYPOTHESIS.

## L2 — ANALYTICAL PRIMITIVES

- ANALYTICAL_CLAIM;
- CALCULATION;
- ASSUMPTION;
- CAUSAL_NODE;
- CAUSAL_EDGE;
- INVALIDATION_TRIGGER;
- ALTERNATIVE_EXPLANATION.

## L3 — ECONOMIC CONTEXT

- COMPANY_ECONOMIC_DNA;
- SECTOR_OVERLAY_ACTIVATION;
- BUSINESS_MODEL_OVERLAY_ACTIVATION.

## L4 — SPECIALIST ENGINE OUTPUTS

- CYCLICALITY;
- TECHNOLOGY;
- INDUSTRY_STRUCTURE;
- MOAT;
- RUNWAY;
- RETURN_QUALITY;
- FCF_FORENSIC;
- CAPITAL_ALLOCATION;
- MANAGEMENT_GOVERNANCE;
- OUTSIDE_VIEW;
- VARIANT_PERCEPTION;
- RISK_RESILIENCE;
- RED_TEAM;
- VALUATION.

## L5 — CANONICAL PROJECTION

- RESEARCH authoritative outputs;
- DEEP_DIVE authoritative outputs;
- deterministic Integration projection.

---

# 4. CORE DESIGN PRINCIPLES

## 4.1 English machine semantics

All canonical keys, enums, IDs and schemas remain English.

The user-facing OroTitan surface translates them into French.

No French label is stored as the canonical state.

## 4.2 No implicit provenance

A material analytical statement may not rely only on surrounding prose.

Material conclusions must expose references to:

- evidence;
- counterevidence;
- calculations;
- assumptions;
- open conflicts;
- invalidation triggers.

## 4.3 No inference presented as fact

Every material evidence/claim item carries an epistemic classification.

Core epistemic types:

```text
FACT
MANAGEMENT_CLAIM
ESTIMATE
ASSUMPTION
INFERENCE
CALCULATION
```

Evidence role is a separate dimension.

## 4.4 Numeric identity

Every material numeric input/output preserves where relevant:

- exact value or range;
- unit;
- currency;
- period;
- as-of date;
- accounting basis;
- source/evidence references;
- transformation/calculation lineage.

A bare `18.4` is not an acceptable material analytical value.

## 4.5 Special states remain explicit

Reuse the existing OroTitan special states where appropriate:

```text
UNKNOWN
NOT_APPLICABLE
NOT_ASSESSABLE
MISSING
NOT_AVAILABLE
```

Do not use null to mean an analytical state.

Null is reserved for structurally optional technical fields when explicitly admitted by schema.

## 4.6 Confidence is not scoring

Any analytical confidence metadata:

- does not change Certification;
- does not override evidence;
- does not create a score;
- does not authorize downstream progression.

## 4.7 Immutable stage artifacts

The authoritative persisted unit remains an immutable artifact version.

Operational UI projections may be normalized for queryability later, but are never a second analytical authority.

---

# 5. IDENTIFIER MODEL

Existing traceability IDs remain string-compatible.

Therefore V2 does **not** require UUID-only evidence/calculation IDs.

Required properties:

- non-empty;
- unique inside the authoritative ledger/version;
- stable when the underlying analytical object is unchanged;
- never silently rebound to different content.

Recommended generated forms:

```text
SRC-...
EVD-...
CFL-...
GAP-...
CLM-...
CALC-...
ASM-...
CAU-N-...
CAU-E-...
INV-...
BLOCK-...
```

Legacy IDs remain valid.

Artifact IDs remain Registry UUIDs.

---

# 6. COMMON ARTIFACT ENVELOPE

Every V2 structured analytical artifact carries:

```text
contract_name
schema_version

run_id
stage
stage_revision

issuer_id
security_id
dossier_id

run_type
canonical_mode
data_cutoff

generated_at
method_version

artifact_role
content
```

The envelope must match the Registry lock.

ChatGPT cannot choose a different cutoff or identity during SAVE.

---

# 7. SOURCE RECORD

A source is distinct from an evidence item.

One source may support multiple evidence items.

Minimum:

```text
source_id
title
publisher
source_class
source_date
data_period
url_or_locator
root_source_id
retrieved_at
access_status
limitations[]
```

Initial source classes:

- REGULATORY_FILING;
- ISSUER_PRIMARY;
- CUSTOMER_PRIMARY;
- COMPETITOR_PRIMARY;
- SUPPLIER_CHANNEL_PRIMARY;
- REGULATORY_INDUSTRY_TECHNICAL_ACADEMIC;
- HIGH_QUALITY_SECONDARY;
- DERIVATIVE_SUMMARY;
- USER_SUPPLIED.

A derivative source should point to the recoverable root source where possible.

---

# 8. EVIDENCE ITEM

Minimum:

```text
evidence_id
source_id
claim_summary
epistemic_type
evidence_role
polarity
affected_blocks[]
source_locator
source_date
data_period
admission_status
limitations[]
```

Evidence roles include:

- FACT;
- MANAGEMENT_CLAIM;
- CUSTOMER_EVIDENCE;
- COMPETITOR_EVIDENCE;
- INDUSTRY_EVIDENCE;
- REGULATORY_EVIDENCE;
- ACADEMIC_TECHNICAL_EVIDENCE;
- CALCULATION;
- ASSUMPTION;
- INFERENCE.

Polarity:

```text
SUPPORTS
CONTRADICTS
CONTEXT
NEUTRAL
```

Admission status:

```text
ADMITTED
PENDING
REJECTED
SUPERSEDED
```

A material Evidence ID used in analysis must resolve to an `ADMITTED` current-run or explicitly admitted lineage item.

---

# 9. CUTOFF VALIDATION

For ordinary current-run evidence:

```text
SOURCE_DATE <= DATA_CUTOFF
```

or a contractually admitted PIT rule must exist.

Post-cutoff evidence:

```text
DO NOT ADMIT
→ FLAG_FOR_REFRESH
```

The validator, not ChatGPT prose, determines cutoff validity.

---

# 10. CONFLICT RECORD

A conflict is first-class.

Minimum:

```text
conflict_id
question_or_claim
conflict_type
side_a_evidence_ids[]
side_b_evidence_ids[]
affected_blocks[]
materiality
status
resolution
resolution_evidence_ids[]
downstream_impact
```

Conflict types:

- FACTUAL;
- TEMPORAL;
- ACCOUNTING_BASIS;
- SCOPE;
- SOURCE_RELIABILITY;
- METHODOLOGICAL;
- INTERPRETIVE.

Status:

- OPEN;
- RESOLVED;
- ACCEPTED_UNCERTAINTY;
- NON_MATERIAL;
- SUPERSEDED.

Material unresolved conflicts remain visible downstream.

---

# 11. OPEN QUESTION / GAP

Canonical analytical shape:

```text
question_id
block
question
why_material
current_evidence_ids[]
missing_evidence
searches_already_performed[]
best_next_source
status
impact_if_unresolved
last_attempt_fingerprint
```

Status:

- OPEN;
- RESOLVED;
- EXHAUSTED;
- NOT_ASSESSABLE;
- NON_MATERIAL.

This maps to `RESEARCH_GAP_REGISTER`.

The UI may label it "Question ouverte".

---

# 12. ANALYTICAL CLAIM

The analytical claim is the central bridge between evidence and block conclusions.

Minimum:

```text
claim_id
block
claim_type
statement
materiality
supporting_evidence_ids[]
contradicting_evidence_ids[]
calculation_ids[]
assumption_ids[]
conflict_ids[]
alternative_explanations[]
invalidation_trigger_ids[]
status
```

Claim types:

- DESCRIPTIVE;
- CAUSAL;
- NORMALIZATION;
- FORECAST;
- COMPARATIVE;
- RISK;
- VALUATION.

Status:

- SUPPORTED;
- MIXED;
- NOT_SUPPORTED;
- LOW_CONFIDENCE;
- NOT_ASSESSABLE.

A material claim cannot be `SUPPORTED` with zero support lineage unless the frozen method explicitly permits a deterministic calculation-only conclusion.

---

# 13. MATERIAL NUMERIC VALUE

Material numeric values use a typed object.

```text
value | range
unit
currency
period_start
period_end
as_of_date
accounting_basis
source_evidence_ids[]
calculation_id
```

Rules:

- exactly one of `value` or `range`;
- range.min <= range.max;
- currency required for monetary values;
- period identity required for flow values;
- as-of date required for stock values where material;
- accounting basis required where IFRS / GAAP / adjusted / economic distinction matters.

---

# 14. CALCULATION

Minimum:

```text
calculation_id
name
formula
method_version
inputs[]
output
precision
evidence_ids[]
assumption_ids[]
reproducibility_status
```

Each input preserves its own period/unit/currency/basis.

Reproducibility status:

- REPRODUCIBLE;
- PARTIAL;
- NOT_REPRODUCIBLE.

Material final calculations used in valuation or scoring must be `REPRODUCIBLE` unless the frozen method explicitly allows a range/unknown state.

---

# 15. ASSUMPTION

Minimum:

```text
assumption_id
block
statement
assumption_type
rationale
evidence_ids[]
sensitivity
affected_claim_ids[]
status
```

Status:

- ACTIVE;
- SUPERSEDED;
- REJECTED.

Assumptions may not masquerade as evidence.

---

# 16. INVALIDATION TRIGGER

Minimum:

```text
trigger_id
statement
affected_claim_ids[]
observable_signal
threshold_or_condition
evidence_ids[]
status
```

Status:

- ACTIVE;
- TRIGGERED;
- RETIRED.

Triggered material invalidation requires revalidation / reopening under the Process Engine.

---

# 17. CAUSAL GRAPH

Nodes reference claims or explicit economic states.

```text
CAUSAL_NODE
node_id
node_type
label
claim_id
evidence_ids[]

CAUSAL_EDGE
edge_id
from_node_id
to_node_id
mechanism
supporting_evidence_ids[]
counterevidence_ids[]
status
```

Edge status:

- SUPPORTED;
- MIXED;
- NOT_SUPPORTED;
- LOW_CONFIDENCE;
- NOT_ASSESSABLE.

A strongly stated material causal conclusion cannot contain an unacknowledged `NOT_SUPPORTED` material edge.

---

# 18. COMPANY ECONOMIC DNA

Company Economic DNA is a Research-stage structured input.

It does not score the company.

Required top-level families:

- sector;
- business-model archetypes;
- revenue/pricing models;
- demand drivers;
- cost structure;
- capital intensity;
- working capital;
- fixed cost intensity;
- reinvestment model;
- organic/acquired growth mix;
- cyclicality;
- technology exposure;
- regulatory exposure;
- geographic exposure;
- customer/supplier concentration;
- distribution;
- installed base;
- network effects;
- intangible intensity;
- M&A dependence;
- commodity exposure;
- leverage model;
- economic bottlenecks;
- key value drivers;
- key failure modes.

Every material DNA field must have field-level provenance.

Field-level provenance uses JSON Pointer:

```text
field_path
supporting_evidence_ids[]
contradicting_evidence_ids[]
assumption_ids[]
```

This keeps the object readable without losing traceability.

---

# 19. OVERLAY ACTIVATION

Overlay activation is explicit and versioned.

```text
overlay_id
overlay_version
overlay_type
activation_reason
supporting_claim_ids[]
status
```

Overlay types are composable.

Examples:

- SOFTWARE;
- MARKETPLACE;
- SERIAL_ACQUIRER;
- SEMICONDUCTOR_EQUIPMENT;
- TECHNOLOGY_BOTTLENECK;
- CYCLICAL;
- INSTALLED_BASE;
- BRAND;
- CREDIT_CYCLE;
- REGULATED.

Status:

- ACTIVE;
- NOT_APPLICABLE;
- REJECTED.

A sector-valid method may not be silently replaced by a generic method.

---

# 20. ANALYTICAL BLOCK OUTPUT ENVELOPE

All specialist engines emit through one common block-output envelope.

```text
block_output_id
block
module_type
module_schema_version
status
summary
material_claim_ids[]
evidence_ids[]
calculation_ids[]
assumption_ids[]
conflict_ids[]
open_question_ids[]
causal_node_ids[]
causal_edge_ids[]
invalidation_trigger_ids[]
dependencies[]
reopened_dependencies[]
module_payload
```

Status:

- IN_PROGRESS;
- BLOCKED;
- COMPLETE;
- NOT_ASSESSABLE;
- REOPENED.

This status is an analytical block state only.

It does not itself complete the Registry stage.

---

# 21. SPECIALIST MODULE PAYLOADS

The common envelope is stable.

Each module payload is separately versioned.

Initial modules:

```text
BUSINESS_MODEL
INDUSTRY_STRUCTURE
TECHNOLOGY
CYCLICALITY
MOAT
RUNWAY
RETURN_QUALITY
FCF_FORENSIC
CAPITAL_ALLOCATION
MANAGEMENT_GOVERNANCE
OUTSIDE_VIEW
VARIANT_PERCEPTION
RISK_RESILIENCE
RED_TEAM
VALUATION
CROSS_BLOCK_RECONCILIATION
```

Do not force every company to populate every optional specialist field.

The overlay and materiality rules determine required depth.

---

# 22. CYCLICALITY PAYLOAD — MINIMUM

Must support:

- cycle types[];
- drivers;
- leading/lagging indicators;
- current regime;
- capacity;
- inventory;
- utilization;
- order book;
- pricing;
- volume;
- margin;
- working capital;
- capex;
- supply response;
- demand response;
- normalized economics;
- peak/trough distortion;
- evidence;
- invalidation triggers.

Where material:

```text
REPORTED_EARNINGS
≠
NORMALIZED_EARNINGS
```

must be represented explicitly.

---

# 23. TECHNOLOGY PAYLOAD — MINIMUM

Must support:

- core technologies;
- architecture;
- bottlenecks;
- proprietary assets/IP;
- standards/interoperability;
- switching costs/workflow embedding;
- R&D model;
- supplier/customer dependencies;
- current/next generation;
- competing/substitute technologies;
- commoditization;
- obsolescence;
- replication difficulty;
- time/capex to replicate;
- certification barriers;
- economic consequences.

Technical complexity alone cannot satisfy moat proof.

---

# 24. MOAT PAYLOAD — MINIMUM

Each moat mechanism records:

- mechanism;
- issuer evidence;
- independent evidence;
- customer evidence;
- competitor evidence;
- behavioral evidence;
- economic consequence;
- evidence against;
- alternative explanation;
- replication path;
- substitution path;
- durability;
- trend.

The payload must reference claims/evidence rather than embed unsupported narrative only.

---

# 25. RUNWAY PAYLOAD — MINIMUM

Must decompose growth into:

- price;
- volume;
- mix;
- share;
- geography;
- product;
- cross-sell;
- installed base;
- market growth;
- M&A;
- cyclical recovery;
- optionality.

Each source identifies:

- contribution type;
- realism constraints;
- required capital;
- marginal economics;
- evidence;
- assumptions.

---

# 26. RETURN QUALITY PAYLOAD — MINIMUM

Supports:

- Standard ROIC;
- All-In ROIC;
- ROIIC;
- marginal returns;
- cohort returns;
- growth-spend economics;
- interpretability;
- attributability.

Near-zero-denominator ROIC must not be treated as automatically superior economics.

---

# 27. FCF / FORENSIC PAYLOAD — MINIMUM

Supports:

```text
REPORTED CASH FLOW
→ ACCOUNTING ADJUSTMENTS
→ ECONOMIC ADJUSTMENTS
→ STANDARDIZED FCF
→ OWNER EARNINGS RANGE
```

Every adjustment records:

- amount/range;
- currency;
- period;
- recurring status;
- cash/non-cash;
- rationale;
- evidence;
- uncertainty.

---

# 28. CAPITAL ALLOCATION PAYLOAD — MINIMUM

Must reconstruct sources/uses over time.

Every material deployment records:

- amount;
- timing;
- funding;
- rationale;
- realized outcome;
- per-share consequence;
- estimated return;
- alternative use;
- evidence.

This is mandatory where M&A or large buybacks materially drive economics.

---

# 29. OUTSIDE VIEW PAYLOAD — MINIMUM

```text
reference_class
prior
comparability
differences
company_specific_evidence
updated_judgment
limitations
```

No generic "base rate" paragraph is sufficient.

---

# 30. RISK PAYLOAD — MINIMUM

Preserve frozen risk taxonomy.

Each risk stores:

- transmission mechanism;
- severity;
- detectability;
- reversibility;
- thesis impact;
- early indicator;
- mitigation;
- invalidation trigger;
- evidence.

---

# 31. RED TEAM PAYLOAD — MINIMUM

Every mandatory perspective records:

- challenge;
- target claims;
- evidence;
- counterargument;
- disposition;
- reopened block IDs.

Disposition:

- REJECTED;
- PARTIALLY_VALID;
- VALID;
- UNRESOLVED.

A `VALID` or material `UNRESOLVED` challenge can force reopening.

---

# 32. VALUATION PAYLOAD — MINIMUM

Keep distinct:

- intrinsic value;
- expected shareholder return;
- market-implied expectations.

Must support:

- primary method;
- cross-checks;
- normalized base;
- bull/base/bear;
- no-multiple-expansion return;
- mature-normalization return;
- reverse valuation;
- sensitivities;
- terminal dependence;
- valuation reliability;
- margin of safety.

Material valuation inputs reference reproducible calculations and explicit assumptions.

---

# 33. CROSS-BLOCK RECONCILIATION

The reconciliation record explicitly checks:

- Technology ↔ Moat;
- Cyclicality ↔ Returns;
- Cyclicality ↔ Valuation;
- Moat ↔ Runway;
- Runway ↔ Reinvestment;
- Return Quality ↔ FCF;
- FCF ↔ Capital Allocation;
- Outside View ↔ Forecast;
- Risk ↔ Valuation;
- Red Team ↔ all affected blocks.

Unresolved material contradictions block normal Certification.

---

# 34. CANONICAL PROJECTION MAP

Research:

```text
SOURCE_RECORD[]
→ RESEARCH_SOURCE_MANIFEST

EVIDENCE_ITEM[]
→ EVIDENCE_LEDGER

CONFLICT_RECORD[]
→ CONFLICT_LEDGER

OPEN_QUESTION[]
→ RESEARCH_GAP_REGISTER

Company Economic DNA / overlays
→ inputs referenced by DD_INPUT_SUFFICIENCY_RECORD
→ persisted as exact supporting analytical artifact(s)
```

Deep Dive:

```text
ANALYTICAL_CLAIM[]
SPECIALIST MODULE OUTPUTS
CAUSAL GRAPH
→ ANALYTICAL_BLOCK_OUTPUTS

CALCULATION[]
→ CALCULATION_LEDGER

ASSUMPTION[]
→ MATERIAL_ASSUMPTION_REGISTER

RED TEAM
→ RED_TEAM_PREMORTEM_RECORD

VALUATION
→ VALUATION_ARTIFACT

Cross-block checks
→ CROSS_BLOCK_RECONCILIATION_RECORD
```

The existing manifest-required artifact list remains unchanged.

---

# 35. SAVE / CHECKPOINT RULE

At CHECKPOINT:

- current structured ledgers may persist as `CHECKPOINT_STAGE_OUTPUT`;
- incomplete block outputs are allowed;
- handoff gate remains NOT_EVALUATED;
- no downstream stage admission.

At FINAL:

- projected required artifacts are complete;
- current versions resolve exactly;
- material references validate;
- cutoff validation passes;
- stage self-audit passes;
- final artifacts are `AUTHORITATIVE_STAGE_OUTPUT`.

---

# 36. VALIDATION CLASSES

Validation runs in this order.

## V1 — JSON STRUCTURE

Schema validity.

## V2 — IDENTITY / LOCK

- run;
- stage;
- revision;
- issuer/security/dossier;
- cutoff;
- method/schema version.

## V3 — REFERENCE INTEGRITY

Every referenced ID resolves to the correct ledger/version.

## V4 — TEMPORAL INTEGRITY

- source cutoff;
- period alignment;
- as-of-date rules.

## V5 — NUMERIC INTEGRITY

- units;
- currency;
- ranges;
- period;
- accounting basis;
- formulas.

## V6 — ANALYTICAL INTEGRITY

Deterministic checks only:

- unsupported material claim reference;
- unresolved material conflict;
- missing required overlay;
- invalid Red Team disposition/reopen relationship;
- invalid dependency status.

No deterministic validator pretends to judge whether a thesis is "smart".

## V7 — STAGE PROJECTION

Required frozen artifacts and handoff semantics.

---

# 37. STORAGE MODEL

Authority:

```text
IMMUTABLE VERSIONED ARTIFACT BYTES
+ REGISTRY
```

Future query/index tables may store projections such as:

- claims;
- evidence index;
- open questions;
- module statuses;
- invalidation triggers.

Those tables are convenience projections.

On conflict:

```text
AUTHORITATIVE ARTIFACT WINS
```

---

# 38. CHATGPT LOAD MODEL

LOAD should not retrieve every V2 object.

Minimum:

```text
RUN LOCK
CURRENT STAGE
CURRENT BLOCK
DEPENDENCIES
RELEVANT CLAIMS
RELEVANT EVIDENCE
MATERIAL CONFLICTS
OPEN QUESTIONS
APPLICABLE DNA / OVERLAYS
```

Other data remains on-demand.

This preserves reasoning quality and context efficiency.

---

# 39. FRENCH-FIRST UI BOUNDARY

Machine:

```text
NOT_ASSESSABLE
OPEN
CYCLICALITY
MATERIAL_CONFLICT
```

UI:

```text
Non évaluable
Ouverte
Cyclicité
Conflit matériel
```

Localization is a presentation concern.

French strings must not become stored canonical enums.

---

# 40. VERSIONING

Each contract has:

- contract name;
- semantic schema version;
- method version where relevant.

Changing labels/UI text:

```text
NO SCHEMA VERSION CHANGE
```

Adding an optional compatible field:

```text
MINOR
```

Changing required meaning, enum semantics or lineage rules:

```text
MAJOR
```

Existing sealed artifacts remain valid under their pinned version.

---

# 41. FIRST IMPLEMENTATION SLICE

Implement and validate first:

1. Common primitives;
2. Evidence Ledger V2;
3. Conflict Ledger V2;
4. Research Gap Register V2;
5. Company Economic DNA V2;
6. Analytical Block Output envelope.

Only then add specialist module payload schemas.

Reason:

> if provenance, reference integrity and block output structure are wrong, every specialist engine will inherit the defect.

---

# 42. ACCEPTANCE REQUIREMENTS

Before this design can be frozen:

- valid fixtures pass;
- missing/invalid references fail;
- post-cutoff evidence fails admission;
- material numeric values without units/period fail where required;
- assumptions cannot be admitted as evidence;
- unsupported material claims fail deterministic completeness;
- invalid block dependencies fail;
- French UI translation does not alter canonical enums;
- legacy Evidence IDs remain representable;
- existing stage manifests remain valid;
- no required frozen artifact is removed or renamed.

---

# 43. STATUS

```text
DATA_CONTRACTS_V2_DESIGN
= DRAFT V0.1

ARCHITECTURE
= FROZEN

PRODUCTION MUTATION
= FALSE

NEXT
= IMPLEMENT CORE JSON SCHEMAS + FIXTURES + VALIDATORS
```
