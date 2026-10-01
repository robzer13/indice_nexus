# OROTITAN ANALYTICAL DATA CONTRACTS V2 — DESIGN V0.1

Status: DESIGN CANDIDATE — NOT FROZEN  
Program: OroTitan Equity Research vNext  
Purpose: machine-readable analytical working artifacts for the ChatGPT-first operating model  
Methodology change: NO  
Scoring change: NO  
Valuation-policy change: NO  
Canonical snapshot change: NO  
V3 economic-share-count authority change: NO

---

# 1. PURPOSE

This design defines the machine-readable contracts required between:

```text
CHATGPT
→ analytical construction

SUPABASE / OROTITAN REGISTRY
→ durable structured state

PROCESS ENGINE
→ validation / dependencies / reopening

OROTITAN UI
→ French-first presentation
```

The contracts exist to make the approved ChatGPT operating protocol executable without turning OroTitan into an analytical generator.

---

# 2. AUTHORITY AND NON-DUPLICATION

This design does not redesign frozen analytical semantics.

Authority order:

1. frozen Analysis Standard;
2. frozen Evidence / Conflict / Calculation / Assumption semantics;
3. pinned V2 / V3 execution-stage contracts;
4. frozen ChatGPT Operating Protocol V1.0;
5. this implementation contract.

Therefore:

```text
EXISTING FROZEN LEDGER SEMANTICS
= REUSED

PARALLEL NEW LEDGER SEMANTICS
= FORBIDDEN
```

The design may choose physical JSON representation, field casing, validation rules and cross-reference mechanics where higher authority leaves them open.

---

# 3. LANGUAGE BOUNDARY

Machine semantics remain English.

```text
JSON KEYS / ENUMS / IDs
= ENGLISH CANONICAL

OROTITAN USER-FACING LABELS
= FRENCH-FIRST
```

The data layer must never contain translated duplicate enums merely for UI display.

---

# 4. CORE DESIGN PRINCIPLES

## 4.1 Traceability first

Every material analytical output must be traceable to:

```text
VERDICT / FINDING
→ EVIDENCE_ID and/or CALCULATION_ID
→ ROOT SOURCE
```

## 4.2 Exact point-in-time context

Every artifact carries:

- RUN_ID;
- ISSUER_ID;
- DATA_CUTOFF;
- schema version;
- artifact type.

Security / dossier identity is carried where applicable.

## 4.3 Unknown preservation

Missing knowledge is represented explicitly.

No schema default may convert missing evidence into:

- zero;
- false;
- neutral;
- unsupported positive;
- arbitrary estimate.

## 4.4 No hidden scalarization

Material numeric values use explicit scalar / range / unknown representation when needed.

V3 economic-share-count semantics remain governed exclusively by the V3 methodology overlay and are referenced, not redefined here.

## 4.5 Cross-object IDs are first-class

Cross-artifact dependencies use exact registry identity:

```text
ARTIFACT_ID
VERSION
RUN_ID
STATUS
SHA256 when available
```

Never resolve a dependency through filename guessing, "latest", or chat memory.

The application must validate:

- Evidence IDs;
- Conflict IDs;
- Calculation IDs;
- Assumption IDs;
- Source IDs;
- Gap / Open Question IDs;
- causal-link IDs;
- block dependencies.

JSON Schema validates shape. OroTitan boundary validators validate referential integrity.

---

# 5. CONTRACT FAMILY

V0.1 defines the following object families:

```text
SOURCE_MANIFEST
EVIDENCE_LEDGER
CONFLICT_LEDGER
CALCULATION_LEDGER
MATERIAL_ASSUMPTION_REGISTER
MATERIAL_RESEARCH_HYPOTHESIS_REGISTER
RESEARCH_GAP_REGISTER
DD_INPUT_SUFFICIENCY_RECORD
ANALYSIS_INPUT_LOCK

COMPANY_ECONOMIC_DNA
INDUSTRY_STRUCTURE_ANALYSIS
ANALYTICAL_BLOCK_OUTPUT
CAUSAL_GRAPH
MOAT_PROOF_ANALYSIS
RUNWAY_ANALYSIS
RETURN_QUALITY_ANALYSIS
FCF_FORENSIC_ANALYSIS
CAPITAL_ALLOCATION_ANALYSIS
OUTSIDE_VIEW_ANALYSIS
VARIANT_PERCEPTION_ANALYSIS
RISK_RESILIENCE_ANALYSIS
RED_TEAM_RECORD
TECHNOLOGY_ANALYSIS
CYCLICALITY_ANALYSIS
MATERIAL_CHANGE_REVALIDATION_RECORD
```

These are working / stage analytical artifacts.

They do not replace the canonical screener snapshot.

---

# 6. FROZEN LEDGER REPRESENTATIONS

## 6.1 Evidence

The representation preserves:

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
```

The following frozen vocabularies are exact and must not be replaced by more conversational labels:

```text
SOURCE_CLASS = S1 | S2 | S3 | S4 | S5 | S6
CLAIM_FIT = HIGH | MEDIUM | LOW
EPISTEMIC_TYPE = REPORTED | CALCULATED | CONSENSUS | ESTIMATE | ASSUMPTION | UNKNOWN
FRESHNESS_STATE = CURRENT | FIT_FOR_PURPOSE | STALE_FOR_USE | UNKNOWN_DATE
```

The ChatGPT operating layer may reason about management claims, customer evidence, competitor evidence or inference, but those concepts must map into the frozen ledger fields rather than redefine `EPISTEMIC_TYPE`.

The frozen Evidence Ledger also preserves `USED_IN[]`.

Additional implementation-only quantitative provenance may attach:

- exact scalar or range;
- unit;
- currency;
- period;
- as-of date;
- accounting basis;
- transformed calculation reference.

It does not alter evidence authority.

## 6.2 Conflict

Preserve exactly the conceptual semantics:

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

## 6.3 Calculation

Preserve:

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

Persisted material numeric results use decimal strings to avoid accidental precision loss.

## 6.4 Material assumption

Preserve:

```text
ASSUMPTION_ID
VARIABLE
VALUE / RANGE
EPISTEMIC_TYPE
SOURCE / RATIONALE
SENSITIVITY
USED_IN
```

---

# 7. RESEARCH EXECUTION OBJECTS

## 7.1 Source Manifest

Preserves the frozen Research conceptual fields and remains broader than the Evidence Ledger.

A discovered source may exist in the Source Manifest without becoming material evidence.

## 7.2 Material Research Hypothesis

A material Research hypothesis is directional research state, never a final Deep Dive verdict.

## 7.3 Research Gap

Preserves:

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

This object is used by the anti-loop system.

## 7.4 DD Input Sufficiency

Preserves the frozen execution-only states:

```text
SUFFICIENT
INSUFFICIENT
NOT_APPLICABLE
```

It does not encode analytical quality.

## 7.5 Analysis Input Lock

Identifies the exact Research state admitted into Deep Dive.

It contains references / versions rather than copies of analytical truth.

---

# 8. COMPANY ECONOMIC DNA

Company Economic DNA is a structured analytical support artifact introduced by Analytical Engine V2.

It does not create scores or frozen verdicts.

Its role is to capture the causal economic architecture of the business:

- who pays;
- what is sold;
- pricing mechanisms;
- volume mechanisms;
- recurring / transactional characteristics;
- cost drivers;
- capital requirements;
- working-capital behavior;
- reinvestment engines;
- acquisition dependency;
- technology dependency;
- regulatory dependency;
- structural bottlenecks;
- key unit economics.

Every material statement must carry traceability references.

The object is intended to feed multiple analytical blocks without forcing each block to rediscover the business architecture.

---

# 9. STANDARD ANALYTICAL BLOCK OUTPUT

Every major analytical block uses one standard execution envelope preserving the frozen Deep Dive structure:

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

Allowed execution status:

```text
INSUFFICIENT
IN_PROGRESS
PROVISIONALLY_STABLE
LOCKED
```

Allowed execution confidence:

```text
HIGH
MEDIUM
LOW
```

Confidence remains execution metadata only.

---

# 10. INDUSTRY STRUCTURE + CAUSAL GRAPH

## 10.1 Industry Structure

Industry Structure is a cross-block support artifact, not a generic Porter paragraph and not a score.

It preserves the Analytical Engine V2 lenses:

```text
MARKET_STRUCTURE
COMPETITOR_SET
MARKET_SHARE_DISTRIBUTION
ENTRY_RATE / EXIT_RATE
CAPACITY_DISCIPLINE
PRICING_DISCIPLINE
CUSTOMER / SUPPLIER / DISTRIBUTOR POWER
REGULATORY_BARRIERS
SWITCHING_FRICTION
MULTIHOMING
VERTICAL_INTEGRATION
CONSOLIDATION_TREND
DISRUPTION_VECTORS
PROFIT_POOL_LOCATION
VALUE_CHAIN_POSITION
HISTORICAL_RETURN_DISTRIBUTION
```

Every material finding remains traceable.

## 10.2 Causal Graph

The Causal Graph does not become a separate source of truth.

It preserves explicit nodes and edges:

```text
CAUSAL_NODE
- NODE_ID
- CLAIM
- TYPE
- EVIDENCE_IDS[]
- CONTRADICTING_EVIDENCE_IDS[]
- CONFIDENCE

CAUSAL_EDGE
- FROM_NODE
- TO_NODE
- MECHANISM
- SUPPORTING_EVIDENCE_IDS[]
- COUNTEREVIDENCE_IDS[]
- STATUS
- CONFIDENCE
```

Allowed edge status:

```text
SUPPORTED
MIXED
NOT_SUPPORTED
LOW_CONFIDENCE
NOT_ASSESSABLE
```

A material causal edge that is unsupported or not assessable cannot be treated as strongly proven merely because the narrative is persuasive.

---

# 11. STRUCTURED BLOCK-SUPPORT ARTIFACTS

For quality-critical analytical engines, structured support artifacts may accompany the canonical `ANALYTICAL_BLOCK_OUTPUT`.

They exist to preserve evidence mechanics and analytical bridges, not to create a second verdict authority.

```text
CANONICAL BLOCK VERDICT
= ANALYTICAL_BLOCK_OUTPUT

STRUCTURED SUPPORT
= ENGINE-SPECIFIC EVIDENCE / BRIDGE / CHALLENGE DATA
```

Supported structured artifacts include:

- `MOAT_PROOF_ANALYSIS`: mechanism → independent/customer/competitor/behavioral evidence → economic consequence → replication/substitution → durability;
- `RUNWAY_ANALYSIS`: causal growth sources plus explicit TAM / serviceable market / realistic capture pool;
- `RETURN_QUALITY_ANALYSIS`: ROIC/ROIIC/marginal/cohort/growth-spend economics with interpretability and attributability;
- `FCF_FORENSIC_ANALYSIS`: explicit adjustment bridge from reported cash flow toward standardized FCF / owner earnings;
- `CAPITAL_ALLOCATION_ANALYSIS`: deployments with amount/timing/funding/outcome/per-share consequence/estimated return/alternative use;
- `OUTSIDE_VIEW_ANALYSIS`: reference class → prior → comparability → company evidence → updated judgment;
- `VARIANT_PERCEPTION_ANALYSIS`: market expectation vs OroTitan view with falsification condition;
- `RISK_RESILIENCE_ANALYSIS`: causal risk records under the frozen taxonomy;
- `RED_TEAM_RECORD`: all mandatory adversarial perspectives and reopening recommendations.

No structured support artifact may contain or override a frozen score.

The block envelope references support artifacts by exact ID/version through the registry layer.

---

# 11. TECHNOLOGY ANALYSIS

Technology Analysis is applicable only when technology is economically material.

It structurally covers:

- current technical architecture;
- technical bottlenecks;
- next-generation path;
- substitution paths;
- replication difficulty;
- supplier dependence;
- customer dependence;
- standards / ecosystem constraints;
- R&D economics;
- commoditization risk;
- economic consequence.

```text
TECHNICAL COMPLEXITY
≠ MOAT
```

No field may mechanically promote technical difficulty into a moat verdict.

---

# 12. CYCLICALITY ANALYSIS

When material, Cyclicality explicitly separates:

- secular growth;
- price effect;
- volume effect;
- inventory cycle;
- capacity cycle;
- end-demand cycle;
- utilization;
- working-capital cycle;
- normalized economics.

The object stores analytical findings and normalization references.

It does not create a valuation result.

---

# 13. CAPITAL ALLOCATION ANALYSIS

The contract must support both ordinary capital allocators and serial acquirers.

Required structured lenses include:

- organic reinvestment;
- acquisitions;
- buybacks;
- dividends;
- debt / deleveraging;
- dilution / SBC where material;
- acquisition economics where material;
- organic-vs-acquired growth bridge where material;
- incremental-return evidence;
- capital-allocation risks.

No acquisition quality score is introduced.

---

# 14. OPEN QUESTIONS

Open Questions must not become a second authoritative register.

The Analytical Engine V2 definition is preserved:

```text
OPEN QUESTIONS
= USER-FACING / WORKBENCH VIEW
OVER
RESEARCH_GAP_REGISTER
+ ANALYTICAL_BLOCK_OUTPUT.UNRESOLVED_POINTS
```

Canonical Research gaps remain in `RESEARCH_GAP_REGISTER`.

Analytical unresolved points remain inside the exact block output that owns them.

The UI projection may classify questions as:

```text
OPEN
RESOLVED
EXHAUSTED
NOT_ASSESSABLE
NON_MATERIAL
```

but persistence of that view must retain source references and may not create competing analytical authority.

---

# 15. MATERIAL CHANGE REVALIDATION

A material change record is mandatory when a previously durable analytical conclusion materially changes.

It stores:

- affected blocks;
- prior state reference;
- proposed new state;
- triggering evidence;
- primary/root-source recheck;
- contradictory research;
- best alternative explanation;
- downstream reconciliation;
- final revalidation outcome;
- rationale.

This object operationalizes the frozen ChatGPT protocol's material-change gate.

It creates no new methodology.

---

# 16. REFERENTIAL-INTEGRITY RULES

JSON Schema alone is insufficient.

Boundary validators must enforce at minimum:

```text
ALL referenced Evidence IDs exist in the active authoritative ledger lineage
ALL Conflict evidence references exist
ALL Calculation input evidence references exist
ALL Assumption used_in targets exist or are declared contract targets
ALL analytical block references use current or explicitly pinned artifact versions
SOURCE_DATE <= DATA_CUTOFF for admitted current-run evidence
ROOT_SOURCE_ID resolves when declared
NO evidence ID appears as both supporting and contradicting the same finding without explicit conflict semantics
LOCKED block cannot contain unresolved material blocker that frozen method says prevents lock
MATERIAL_CHANGE_REVALIDATION must complete before final-sealing a material changed conclusion
```

---

# 17. PERSISTENCE MODEL

Normal persistence remains artifact-first.

```text
JSON ARTIFACT BODY
→ schema validation
→ semantic / referential validation
→ immutable bytes
→ SHA-256
→ Registry artifact
→ manifest
```

These contracts do not require one SQL table per object family.

Supabase physical representation is a later implementation decision.

---

# 18. CHATGPT OPERATING MODEL

ChatGPT may freely construct working reasoning.

Only SAVE / CHECKPOINT converts accepted work into these structured contracts.

```text
CHAT_WORKING
→ not required to validate continuously

CHECKPOINT / SAVE
→ must produce valid contract objects
```

This preserves maximum reasoning quality while keeping durable state strict.

---

# 19. UI PROJECTION

The OroTitan UI is French-first.

Examples:

```text
OPEN_QUESTION → Question ouverte
CONFLICT → Conflit
SUPPORTING_EVIDENCE → Preuves favorables
CONTRADICTING_EVIDENCE → Preuves contradictoires
REOPEN_TRIGGER → Déclencheur de réouverture
```

Translations belong to the presentation layer.

Canonical stored enums remain English.

---

# 20. V3 COMPATIBILITY

V3 economic-share-count methodology remains a separate global authority.

This Data Contracts V2 design:

- may reference V3 denominator artifacts / calculations;
- may carry those references into Valuation-related blocks;
- must never redefine EXACT / BOUNDED / UNKNOWN denominator methodology;
- must never midpoint a V3 bound;
- must never alter V3 admission rules.

---

# 21. ACCEPTANCE TARGETS CLOSED BY THIS DESIGN

This contract family directly targets protocol-battery gaps:

```text
T04 POST_CUTOFF_SOURCE
T05 ISSUER_COMPETITOR_CONTRADICTION
T07 PRIOR_CANONICAL_CONCLUSION_OVERTURNED
T10 SERIAL_ACQUIRER
T11 WRONG_SECTOR_METHOD_ATTEMPTED (data support only; gate belongs to Process Engine)
T12 MATERIAL_GAP_UNRESOLVED (data support only; state transition belongs to Process Engine)
T16 INVALID_EVIDENCE_ID
T18 RED_TEAM_REOPENS_EARLIER_BLOCK (data support only)
T20 ROUTINE_FUNDAMENTAL_DELTA (affected-block data support)
```

Process behavior itself remains for Process Engine V2.

---

# 22. FILES

Candidate machine schema:

`schemas/vnext/orotitan-analytical-data-contracts-v2.schema.v0.1.json`

Candidate tests:

`tests/vnext-analytical-data-contracts-v2-design.test.ts`

---

# 23. DESIGN STATUS

```text
DOCUMENT = OROTITAN_ANALYTICAL_DATA_CONTRACTS_V2_DESIGN_V0.1
STATUS = DESIGN_CANDIDATE
FROZEN = NO
PRODUCTION_MUTATION = NO
CANONICAL_SNAPSHOT_CHANGE = NO
NEXT_REVIEW = SCHEMA_AND_ACCEPTANCE_TESTS
```
