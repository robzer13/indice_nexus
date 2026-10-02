# OroTitan VNext Analytical Data Contracts

Status: GLOBAL REVIEW PATCHED — CANDIDATE FOR FREEZE

This directory is the single current Data Contracts V2 candidate package for the ChatGPT-first Analytical Engine V2. The earlier consolidated post-C7 schema is retained only as superseded historical design material.

## Authority boundary

These schemas implement existing frozen analytical semantics.

They do not replace:

- Analysis Standard V1;
- Research / Deep Dive / Integration Stage Contracts;
- Research Execution Process;
- the live OroTitan artifact registry;
- the Phase-4 Screener projection schema;
- frozen Gate-15 deterministic modules.

## Core artifact contracts

- `research-source-manifest`
- `evidence-ledger`
- `conflict-ledger`
- `calculation-ledger`
- `material-assumption-register`
- `material-research-hypothesis-register`
- `research-gap-register`
- `dd-input-sufficiency-record`
- `analysis-input-lock`
- `analytical-block-output`

## Adaptive analytical support contracts

- `company-economic-dna`
- `overlay-selection`
- `cycle-analysis`
- `technology-map`
- `industry-structure`
- `causal-graph`
- `serial-acquirer-capital-deployment`

Adaptive support artifacts feed canonical analytical blocks. They do not create alternate scoring, Certification or OroTitan terminal semantics.

## Validation layers

1. JSON Schema: structure and frozen vocabulary.
2. Cross-artifact validator: cutoff, identity, versions and references.
3. Process Engine V2: reopening, material-change revalidation, routing and stage transitions.
4. Integration: deterministic projection into the existing canonical snapshot.

## Language

Persisted machine vocabulary remains English.

The OroTitan product surface is French-first and maps canonical machine states to French display labels.


## Semantic separation added by global review

The package preserves two distinct layers:

```text
CANONICAL epistemic_type
= REPORTED | CALCULATED | CONSENSUS | ESTIMATE | ASSUMPTION | UNKNOWN

PROTOCOL working_claim_type
= FACT | MANAGEMENT_CLAIM | ESTIMATE | ASSUMPTION | INFERENCE | CALCULATION
```

These fields are orthogonal. The protocol classification must never overwrite the canonical frozen epistemic type.

Likewise:

```text
ANALYTICAL block execution status
= INSUFFICIENT | IN_PROGRESS | PROVISIONALLY_STABLE | LOCKED

PROCESS ENGINE control / reopen state
= separate future process-layer semantics
```

Process Engine V2 must not replace the frozen analytical execution-status vocabulary.
