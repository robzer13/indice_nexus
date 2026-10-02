# OROTITAN_ANALYTICAL_DATA_CONTRACTS_V2_GLOBAL_REVIEW_2026-10-02

**Project:** OroTitan Equity Research  
**Status:** PASS — CI GREEN — READY FOR FREEZE  
**Baseline:** `vnext@ebae407911c4ed4b29f3006f646548695c9747dc`  
**Methodology change:** NO  
**Production mutation:** NONE

## 1. Why this global review was required

PR #330 closed the review of the first post-C7 consolidated Data Contracts candidate. PR #333 subsequently added the split Analytical Engine V2 schema package, and PR #334 amended its analytical block vocabulary.

The two candidates therefore coexisted on `vnext` and were not semantically identical.

## 2. Material findings

### DCV2-R01 — dual authority

Two Data Contracts V2 representations existed:

- `contracts/orotitan-equity/post-c7/OROTITAN_ANALYTICAL_DATA_CONTRACTS_V2_SCHEMA_V0.1.json`;
- `schemas/vnext/data-contracts/*`.

A future Process Engine could silently bind to the wrong vocabulary.

### DCV2-R02 — epistemic semantic conflation

The consolidated candidate used:

```text
FACT | MANAGEMENT_CLAIM | ESTIMATE | ASSUMPTION | INFERENCE | CALCULATION
```

as `epistemic_type`.

The frozen analytical evidence vocabulary remains:

```text
REPORTED | CALCULATED | CONSENSUS | ESTIMATE | ASSUMPTION | UNKNOWN
```

The ChatGPT protocol classification is useful, but it is a distinct working claim classification and must not replace the canonical epistemic type.

### DCV2-R03 — block lifecycle semantic conflation

The consolidated candidate used process-like states such as `READY`, `CHECKPOINTED`, `REOPENED` and `STALE` as the analytical block status.

The split package preserves the analytical execution vocabulary:

```text
INSUFFICIENT | IN_PROGRESS | PROVISIONALLY_STABLE | LOCKED
```

Process Engine V2 may define separate deterministic control/reopen state, but it must not rewrite the analytical execution status.

### DCV2-R04 — free-form analytical block references

`overlay-selection.affected_blocks` and `material-research-hypothesis-register.affected_analytical_block` were free strings even though the analytical block namespace is now canonical.

## 3. Resolution

The split package under:

```text
schemas/vnext/data-contracts/
runtime/vnext/analytical-data-contracts.ts
runtime/vnext/adaptive-analytical-data-contracts.ts
```

is the single current Data Contracts V2 candidate for freeze.

The prior consolidated schema is retained as:

```text
SUPERSEDED_HISTORICAL_DESIGN_CANDIDATE
```

and must not be used as authority by Process Engine V2.

The split Evidence Ledger now preserves both distinct dimensions:

```text
epistemic_type
= canonical analytical semantics

working_claim_type
= ChatGPT protocol working classification
```

It also carries `polarity` and canonical `affected_blocks[]` for material evidence.

## 4. Deferred intentionally to Process Engine V2

This review does not move process semantics into the analytical contracts. The following remain Process Engine scope:

- material-change revalidation gate execution;
- dependency-cone reopening;
- refresh routing;
- blocker-aware next-action resolution;
- retry / loop prevention;
- checkpoint/finalization eligibility.

## 5. Freeze gate

```text
GLOBAL_DATA_CONTRACTS_REVIEW
= PASS

VNext CI
= PASS

Screener CI
= PASS

CODE REVIEW
= P1 FOUND → FIXED → THREAD RESOLVED

FREEZE
= AUTHORIZED

NEXT
= FREEZE_ANALYTICAL_DATA_CONTRACTS_V2
```
