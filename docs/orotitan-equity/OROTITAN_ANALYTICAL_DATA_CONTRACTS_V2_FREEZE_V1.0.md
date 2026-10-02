# OROTITAN_ANALYTICAL_DATA_CONTRACTS_V2_FREEZE_V1.0

**Project:** OroTitan Equity Research  
**Status:** FROZEN — V1.0  
**Freeze date:** 2026-10-02  
**Reviewed baseline:** `vnext@c956cfbadeeaab0b36c01e242a27487fb7a433ef`  
**Review PR:** #336  
**Methodology change:** NO  
**Scoring change:** NO  
**Valuation change:** NO  
**Production mutation:** NONE

## 1. Frozen authority

The authoritative Analytical Data Contracts V2 package is:

```text
schemas/vnext/data-contracts/
runtime/vnext/analytical-data-contracts.ts
runtime/vnext/adaptive-analytical-data-contracts.ts
```

The earlier consolidated candidate under:

```text
contracts/orotitan-equity/post-c7/
lib/orotitan-equity/post-c7/analytical-data-contracts-v2.ts
```

is historical provenance only and is not authority for new Process Engine V2 work.

## 2. Frozen semantic separations

### 2.1 Canonical evidence epistemics vs protocol working claim classification

```text
CANONICAL epistemic_type
= REPORTED
| CALCULATED
| CONSENSUS
| ESTIMATE
| ASSUMPTION
| UNKNOWN
```

remains distinct from:

```text
PROTOCOL working_claim_type
= FACT
| MANAGEMENT_CLAIM
| ESTIMATE
| ASSUMPTION
| INFERENCE
| CALCULATION
```

Neither namespace may silently replace or coerce the other.

### 2.2 Analytical block execution vs Process Engine control state

The analytical block execution vocabulary remains:

```text
INSUFFICIENT
| IN_PROGRESS
| PROVISIONALLY_STABLE
| LOCKED
```

Process Engine V2 may define operational control, freshness, reopen or retry state in its own layer, but those states must not be written into the analytical `execution_status` field.

### 2.3 Canonical analytical block namespace

```text
BUSINESS_MODEL
ECONOMIC_QUALITY
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
RISK_RESILIENCE
RED_TEAM
VALUATION
CROSS_BLOCK_RECONCILIATION
```

Material evidence, analytical block outputs, overlay affected blocks and material research hypotheses must use this namespace where they reference analytical blocks.

## 3. Validation state at freeze

```text
GLOBAL REVIEW
= PASS

VNext CI
= PASS

Screener CI
= PASS

Codex P1
= FOUND
→ FIXED
→ RESOLVED

DUAL DATA-CONTRACT AUTHORITY
= CLOSED
```

## 4. Deliberately not frozen here

This contract does not define Process Engine V2 behavior for:

- material-change revalidation execution;
- dependency-cone reopening;
- refresh routing;
- retry / loop prevention;
- blocker-aware next action;
- checkpoint vs final sealing eligibility;
- ChatGPT ↔ Supabase LOAD / SAVE transport.

Those are downstream execution-layer responsibilities.

## 5. Change control

Any semantic change to the frozen Data Contracts V2 package requires an explicit successor version.

No downstream layer may silently:

- replace canonical epistemic vocabularies;
- overload analytical execution status with process state;
- introduce free-form analytical block identifiers where the canonical namespace applies;
- promote the superseded consolidated candidate back to authority.

## 6. Next exact action

```text
DESIGN_PROCESS_ENGINE_V2
```

Vertical slice remains blocked until Process Engine V2 and the controlled ChatGPT ↔ Supabase bridge are completed and tested.
