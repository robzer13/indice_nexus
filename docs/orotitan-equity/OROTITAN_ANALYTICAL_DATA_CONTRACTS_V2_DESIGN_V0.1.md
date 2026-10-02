# OROTITAN ANALYTICAL DATA CONTRACTS V2 — DESIGN V0.1

Status: SUPERSEDED DESIGN CANDIDATE — HISTORICAL ONLY  
Program: OroTitan Equity Research vNext / post-C7  
Depends on:
- OROTITAN_CHATGPT_OPERATING_PROTOCOL_V1_FREEZE_V1.0
- OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0
- OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2_FREEZE_V2.0
- OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0

Methodology change: NO  
Scoring change: NO  
Valuation change: NO  
Production mutation: NO

## Authority notice

This consolidated design candidate is retained for historical provenance only. After PR #333 and PR #334, the current Data Contracts V2 candidate package is `schemas/vnext/data-contracts/` plus the `runtime/vnext` validators. New Process Engine V2 work must not use the consolidated candidate as semantic authority.

---

## 1. Purpose

These contracts define the machine-readable working state required for a high-quality ChatGPT-led analytical workflow.

They do not replace the frozen Evidence / Conflict / Calculation / Assumption semantics.

They provide a strict composable layer so that ChatGPT can:

- load only relevant context;
- trace every material conclusion;
- preserve uncertainty;
- separate facts, management claims, assumptions, inference and calculations;
- reopen only affected blocks;
- save without polluting canonical state;
- resume from exact persisted state.

The core chain remains:

```text
SOURCE
→ EVIDENCE
→ CONFLICT / GAP / ASSUMPTION
→ CAUSAL LINK
→ ANALYTICAL BLOCK
→ MATERIAL CHANGE REVALIDATION
→ STAGE ARTIFACTS
```

---

## 2. Design principles

### 2.1 No monolithic analysis blob

A long narrative report is not the primary machine contract.

Narrative may be generated from structured analytical state, but cannot replace it.

### 2.2 Existing authorities remain authoritative

This design reuses existing:

- Evidence Grade values;
- Evidence Ledger lineage;
- Conflict Ledger lineage;
- Material Assumption semantics;
- Stage Manifest / Artifact Registry;
- immutable artifact versioning;
- DATA_CUTOFF;
- run identity.

### 2.3 ChatGPT reasoning remains flexible

The contract structures durable outputs only.

It does not force ChatGPT to reason in a rigid form during exploration.

### 2.4 Unknown is first-class

Valid durable states include:

- BLOCKED;
- NOT_ASSESSABLE;
- unresolved conflict;
- exhausted research gap;
- mixed causal link;
- unknown source independence/freshness.

The contract must never force false precision.

---

## 3. Core objects

### SOURCE_RECORD

Stores provenance and cutoff-relevant source metadata.

Minimum:
- SOURCE_ID
- title
- publisher
- type
- SOURCE_DATE
- AS_OF_DATE where applicable
- locator
- ROOT_SOURCE_ID where derivative
- access status
- limitations
- optional content hash

### EVIDENCE_RECORD

Represents one material claim tied to one source.

Separates:
- FACT
- MANAGEMENT_CLAIM
- ESTIMATE
- ASSUMPTION
- INFERENCE
- CALCULATION

Preserves frozen evidence grades:

```text
E0 CLAIM_ONLY
E1 OBSERVABLE_ISSUER_EVIDENCE
E2 STRONG_INDEPENDENT_EVIDENCE
E3 ORTHOGONALLY_CORROBORATED
EU UNKNOWN / CONFLICTED
```

### NUMERIC_DATUM

Material numbers preserve:
- metric;
- scalar/range;
- unit;
- currency;
- period/as-of;
- accounting basis;
- transformation;
- calculation link.

This prevents silent comparison of mismatched:
- currencies;
- periods;
- accounting bases;
- reported vs adjusted metrics.

### CONFLICT_RECORD

Explicitly links contradictory evidence.

A conflict cannot disappear through narrative synthesis.

### GAP_RECORD

Prevents research loops.

Stores:
- exact question;
- affected block;
- materiality;
- searches already performed;
- best next source;
- why unresolved;
- impact;
- state.

### ASSUMPTION_RECORD

Assumptions are explicitly separate from facts/evidence.

### COMPANY_ECONOMIC_DNA

A reusable factual/economic profile, not a score.

Dimensions include:
- revenue engine;
- customers;
- suppliers;
- cost structure;
- capital intensity;
- reinvestment model;
- pricing power mechanism;
- switching costs;
- network effects;
- scale;
- regulation;
- technology dependency;
- cyclicality;
- acquisition dependency;
- geography.

Every dimension may remain null/unknown but retains evidence references.

### CAUSAL_LINK

Formalizes:

```text
CAUSE / MECHANISM
→ ECONOMIC CONSEQUENCE
```

States:
- SUPPORTED
- MIXED
- UNPROVEN
- INVALIDATED

### ANALYTICAL_BLOCK_OUTPUT

Represents durable block state.

Status values:

```text
READY
IN_PROGRESS
CHECKPOINTED
COMPLETE
BLOCKED
NOT_ASSESSABLE
REOPENED
STALE
```

Contains:
- conclusion;
- supporting evidence;
- counterevidence;
- conflicts;
- gaps;
- assumptions;
- causal links;
- applicable sector overlays;
- invalidation triggers;
- upstream block references.

### MATERIAL_CHANGE

A material analytical change cannot be finalized from conversational momentum.

A PASS requires all:

```text
reopened material evidence
verified primary/root sources
searched disconfirming evidence
tested best alternative explanation
reconciled downstream blocks
recorded reason prior state changed
```

### SERIAL_ACQUIRER_PROFILE

Only material when applicable.

Requires explicit treatment of:
- organic vs acquired growth;
- purchase-price discipline;
- integration model;
- goodwill/intangibles;
- earnouts/contingent consideration where material;
- dilution/funding;
- acquisition ROI;
- deployment runway.

---

## 4. Semantic validation beyond JSON Schema

JSON Schema checks shape.

The semantic validator checks relationships.

### Mandatory semantic controls

1. Run identity must match authoritative context.
2. DATA_CUTOFF must match the run.
3. SOURCE_DATE must not exceed DATA_CUTOFF. DATA_PERIOD / AS_OF_DATE describe the datum and may be future when the source was already available by cutoff.
4. Evidence SOURCE_ID must exist.
5. ROOT_SOURCE_ID must resolve, must not self-reference and must not form a cycle.
6. Evidence/conflict/gap/assumption/material-change IDs must be unique.
7. Analytical block code must be unique within one package.
8. All Evidence IDs used by blocks/causal links/overlays must resolve and declare matching block relevance.
9. Conflict IDs / Gap IDs / Assumption IDs must resolve.
10. Numeric range min <= max.
11. Numeric period start <= end.
12. Numeric datum must have a temporal anchor.
13. COMPLETE block must carry traceable evidence.
14. COMPLETE block cannot carry an OPEN / UNRESOLVED_BLOCKING conflict.
15. COMPLETE block cannot carry a critical unresolved or exhausted-NOT_ASSESSABLE gap.
16. SUPPORTED causal links require evidence; MIXED causal links require evidence and counterevidence.
17. Causal link IDs and sector-overlay names must be unique within a block.
18. NOT_ASSESSABLE requires traceable exhausted/blocked gap.
19. REQUIRED_MISSING sector overlay prevents COMPLETE.
20. Block cannot depend on itself.
21. Upstream block references must exist in the package.
22. Material revalidation PASS requires all frozen checklist controls.
23. Applicable serial acquirer profile requires actual acquisition-economics analysis.

---

## 5. Mapping to regression-battery gaps

### Closed / materially addressed by this package

```text
POST_CUTOFF_EVIDENCE_VALIDATOR
V2_EVIDENCE_AND_CONFLICT_CONTRACTS
ANALYTICAL_EVIDENCE_ID_VALIDATION
SERIAL_ACQUIRER_CAPITAL_ALLOCATION_STRUCTURES
BLOCK_LEVEL_GAP_AND_NOT_ASSESSABLE_STATE
MATERIAL_CHANGE_REVALIDATION_GATE — data representation + validator portion
SECTOR_OVERLAY_SELECTION_VALIDATION — data representation + block guard portion
BLOCK_LEVEL_DEPENDENCY_REOPENING — dependency representation portion
```

### Intentionally left to Process Engine V2

```text
BLOCK_LEVEL_DEPENDENCY_REOPENING — transition algorithm
PRICE_ONLY_DELTA_ROUTING
ROUTINE_FUNDAMENTAL_DELTA_ROUTING
FULL_REFRESH_ROUTING
STAGE_FINALIZATION_ELIGIBILITY_FROM_SAVE
MATERIAL_CHANGE_REVALIDATION_GATE — transition enforcement
SECTOR_OVERLAY_SELECTION_VALIDATION — overlay-selection engine
```

### Intentionally left to ChatGPT ↔ Supabase bridge

```text
LOAD_RESOLVER_AND_CONTEXT_ASSEMBLER
SAVE_ORCHESTRATION
CONNECTOR_FAILURE_FAIL_CLOSED_LOAD_PATH
bounded guarded-RPC invocation
```

This separation prevents contracts from becoming an orchestration engine.

---

## 6. French-first product boundary

Canonical machine fields and enums remain English.

The future OroTitan product surface translates them into French.

Example:

```text
NOT_ASSESSABLE
→ Non évaluable

BLOCKED
→ Bloqué

OPEN
→ Ouvert

EXHAUSTED_NOT_ASSESSABLE
→ Recherche épuisée — non évaluable
```

Translation belongs to the presentation layer, not the canonical analytical contract.

---

## 7. Persistence model

These objects are not required to become new database tables one-for-one.

Preferred persistence remains:

```text
immutable artifact bytes
+ Artifact Registry
+ exact artifact references
+ stage manifests
```

The eventual implementation may expose indexed projections for UI/query performance without changing artifact authority.

No new live Supabase DDL is authorized by this design candidate.

---

## 8. Compatibility rule

The schema is an overlay for post-C7 working-state quality.

It must not:
- rewrite historical V1/V2/V3 artifacts;
- alter scoring formulas;
- alter valuation methodology;
- create a new Evidence Ledger authority;
- replace Stage Manifest semantics;
- bypass Certification;
- bypass GO PUBLISH.

---

## 9. Acceptance criteria

Before this contract is eligible to freeze:

1. JSON Schema parses under AJV 2020.
2. coherent package validates.
3. post-cutoff SOURCE_DATE fails closed.
4. forward estimate / guidance periods known at cutoff remain valid.
5. unknown Evidence ID fails closed.
6. evidence used by a block must declare matching block relevance.
7. duplicate analytical block code fails closed.
8. invalid root-source self-reference / cycle fails closed.
9. open/blocking conflict blocks COMPLETE.
10. critical unresolved or exhausted-NOT_ASSESSABLE gap blocks COMPLETE.
11. COMPLETE block requires traceable evidence.
12. SUPPORTED / MIXED causal links enforce evidence roles.
13. duplicate causal-link IDs / sector overlays fail closed.
14. NOT_ASSESSABLE requires traceable gap.
15. incomplete revalidation cannot PASS.
16. applicable serial acquirer profile cannot be empty.
17. numeric datum requires period/as-of.
18. absent upstream block fails.
19. REQUIRED_MISSING overlay blocks COMPLETE.
20. frozen Evidence Grade values remain exact.
21. no production DB mutation is required.
22. no existing frozen contract is modified.

---

## 10. Current files

```text
contracts/orotitan-equity/post-c7/
  OROTITAN_ANALYTICAL_DATA_CONTRACTS_V2_SCHEMA_V0.1.json

lib/orotitan-equity/post-c7/
  analytical-data-contracts-v2.ts

tests/
  vnext-analytical-data-contracts-v2-design.test.ts
```

---

## 11. Status

```text
DOCUMENT = OROTITAN_ANALYTICAL_DATA_CONTRACTS_V2_DESIGN_V0.1
STATUS = DESIGN_CANDIDATE_REVIEW_PATCHED
FROZEN = NO
PRODUCTION_MUTATION = NO
NEXT = REVIEW_FIX_CI + FINAL DESIGN REVIEW
```
