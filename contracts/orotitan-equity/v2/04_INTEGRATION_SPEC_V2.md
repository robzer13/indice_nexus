# 04_INTEGRATION_SPEC_V2

**Status:** FROZEN IMPLEMENTATION AUTHORITY — V2.0
**Base analytical projection:** `04_INTEGRATION_SPEC_V1_PATCHED`
**Base deterministic computation:** `I2_CANONICAL_COMPUTATION` unchanged
**Publication writer:** V2 validated snapshot writer

## 1. Purpose

This specification is the V2 successor for canonical integration. It does not change the frozen analytical methodology, scoring formulas, valuation conventions, semantic states or terminal OroTitan gate. It adds the V2 product projection and a separate V2 persistence boundary while preserving V1 snapshots and the V1 writer.

## 2. Validation composition

A V2 canonical snapshot MUST pass both layers:

```text
LAYER A — V1 CORE
04_SCREENER_SCHEMA_V1_PATCHED
+ existing semantic-state checks
+ existing I2 deterministic reconciliation
+ existing history-transition checks

LAYER B — V2 PRODUCT EXTENSION
04_SCREENER_SCHEMA_V2
+ OROTITAN_TAXONOMY_V2.0
+ business description rules
+ structured investment thesis rules
+ descriptive PEA rules
```

The V2 validator removes only the top-level `v2_product` member before delegating Layer A to the unchanged V1 validator. It then validates the full `v2_product` object under Layer B. No V1 analytical field is reconstructed from V2 product metadata.

## 3. Canonical V2 projection

V2 snapshots add exactly one top-level object:

```text
v2_product
  classification
    issuer_country_code
    primary_listing_country_code
    sector
    industry_group
    business_model_primary
    business_model_secondary
    economic_exposure_regions[]
    taxonomy_version

  business_summary
    business_description_short

  investment_thesis
    quality_case
    valuation_case
    key_risk

  portfolio_filters [optional]
    pea_eligibility
    pea_eligibility_as_of
    pea_eligibility_source_ref
```

`classification`, `business_summary` and `investment_thesis` are required for a V2 ANALYZE/REFRESH publication candidate. `portfolio_filters` is descriptive only.

## 4. Taxonomy authority

`OROTITAN_TAXONOMY_V2.0.json` is the sole controlled-list authority for sector, industry group, business model, economic exposure region and PEA state. Country codes must be valid ISO 3166-1 alpha-2 codes.

Taxonomy has zero scoring authority. Integration MUST NOT derive or modify OQS, OVS, Investment Score, terminal state, valuation assumptions or fundamental judgments from classification values.

## 5. Business description

`business_description_short` MUST:

```text
be 1–450 characters
be factual and neutral
state what is sold, to whom and how money is made
contain no score language
contain no valuation conclusion
contain no buy/sell recommendation
```

## 6. Structured thesis

Certification owns the analytical content. Integration only maps the exact certified values:

```text
quality_case
valuation_case
key_risk
```

Each field is required and limited to 240 characters. Integration may not rewrite or improve them.

## 7. PEA descriptive state

If `pea_eligibility` is `YES` or `NO`, both an as-of date and source reference are required. `UNKNOWN` may omit them. PEA state has no scoring or terminal-gate authority.

## 8. Persistence version lock

V2 publication candidates use:

```text
contract_version = 04_SCREENER_SCHEMA_V2
schema_version = 2.0.0
```

V1 snapshots remain:

```text
contract_version = 04_SCREENER_SCHEMA_V1
schema_version = 1.0.0
```

The database accepts only these two exact pairs. Mixed pairs are invalid.

The V1 function `persist_orotitan_research_snapshot(uuid, uuid, jsonb)` remains V1-only. V2 uses the additive function `persist_orotitan_research_snapshot_v2(uuid, uuid, jsonb)`.

## 9. Publication boundary

Integration may prepare a V2 canonical snapshot candidate and a `READY_TO_PUBLISH = YES` state. It MUST NOT advance the dossier pointer until separate user authorization:

```text
GO PUBLISH <COMPANY>
```

## 10. V1 compatibility

Existing V1 snapshots remain immutable and renderable. No migration adds V2 product fields to historical V1 payloads. Public/read-model code MUST treat `v2_product` as optional so V1 snapshots remain valid historical product truth.

## 11. Fail-closed conditions

Block V2 admission/persistence on any of:

```text
V1 core schema failure
V2 overlay schema failure
invalid controlled taxonomy value
invalid ISO country code
invalid business description
invalid structured thesis
inconsistent PEA evidence state
I2 reconciliation failure
history transition failure
V1/V2 contract-schema pair mismatch
writer privilege mismatch
snapshot pointer concurrency conflict
```
