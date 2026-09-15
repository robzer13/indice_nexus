# OROTITAN_INTEGRATION_STAGE_CONTRACT_V2 — FREEZE V2.0

**Status:** FROZEN DESIGN — V2.0  
**Depends on:** V2 Process + V2 Pilotage + V2 Deep Dive  
**Methodology change:** NO  
**I2 / I3-B change:** NO

## 0. Purpose

Integration converts the FINAL admitted Deep Dive state into a schema-valid, deterministic, traceable canonical snapshot candidate. Integration is not an analytical stage and may not rewrite upstream judgments.

## 1. Admission

Integration may start only if authoritative persisted state proves:

```text
DEEP_DIVE_STAGE_STATUS = COMPLETE
READY_FOR_INTEGRATION = YES
active Deep Dive manifest = FINAL
exact final Deep Dive artifact set = available + hash-verified
RUN_ID / issuer / security / dossier / DATA_CUTOFF = consistent
contract pins = exact
```

A Fundamentals or Valuation CHECKPOINT manifest never admits Integration.

## 2. Responsibilities

Integration owns only:

```text
canonical mapping
V2 canonical product projection
schema validation
semantic-state validation
I2 deterministic reconciliation
history transition validation
I3-B admission
canonical snapshot candidate construction
pre-publication control card
FINAL Integration Stage Manifest
READY_TO_PUBLISH
```

Integration MUST NOT:

```text
redo Research
reinterpret Fundamentals
change a dimension judgment
change a valuation assumption
repair scoring inputs by choosing alternative values
invent taxonomy or thesis content absent from admitted Deep Dive artifacts
use a human summary as authority
```

Analytical contradiction -> fail closed -> exact upstream reopen route.

## 3. V2 canonical projection

The candidate must project, in addition to existing frozen analytical content:

```text
classification.issuer_country_code
classification.primary_listing_country_code
classification.sector
classification.industry_group
classification.business_model_primary
classification.business_model_secondary
classification.economic_exposure_regions
classification.taxonomy_version
business_summary.business_description_short
investment_thesis.quality_case
investment_thesis.valuation_case
investment_thesis.key_risk
portfolio_filters.pea_eligibility where available
portfolio_filters.pea_eligibility_as_of where available
```

These fields are mapped from exact admitted artifacts. Integration cannot create their semantics.

## 4. I2 deterministic reconciliation

I2 remains exact and fail-closed.

Persisted full-precision scoring inputs must deterministically reproduce the admitted Deep Dive score outputs under the pinned I2 authority.

Mismatch:

```text
I2_RECONCILIATION = FAIL
READY_TO_PUBLISH = NO
NO SILENT REWRITE
ROUTE UPSTREAM
```

Display rounding is not a reconciliation tolerance.

## 5. I3-B admission

I3-B remains unchanged and fail-closed. Broken traceability, invalid state mapping, unsupported special state, score-permission violation or schema inconsistency blocks admission.

## 6. Required Integration artifacts

Persist/version at minimum:

```text
CANONICAL_SNAPSHOT_CANDIDATE
INTEGRATION_MAPPING_RECORD
SCHEMA_VALIDATION_REPORT
I2_RECONCILIATION_REPORT
HISTORY_TRANSITION_VALIDATION_REPORT
I3B_ADMISSION_REPORT
PRE_PUBLICATION_CONTROL_CARD
INTEGRATION_STAGE_MANIFEST
```

## 7. Completion

Integration may become `COMPLETE` only after:

```text
mapping complete
schema validation PASS
semantic-state integrity PASS
I2 reconciliation PASS
history transition PASS / NOT_APPLICABLE as valid
I3-B admission PASS
candidate bytes persisted and hash-verified
Registry reconciled
FINAL Integration Stage Manifest registered
READY_TO_PUBLISH = YES
```

No production promotion is implied.

## 8. Publication boundary

Only explicit:

```text
GO PUBLISH <COMPANY>
```

may authorize canonical promotion of the exact admitted candidate associated with the exact FINAL Integration Stage Manifest.

After successful promotion, the dossier `current_snapshot_id` advances to the new immutable snapshot and prior snapshots remain historical.

## 9. Final user-facing handoff

If `READY_TO_PUBLISH = YES`, the final visible block is exactly the publication command for the company. No prose follows it.

If blocked, the final visible block is the exact resolution/Pilotage prompt. No publication command is emitted.
