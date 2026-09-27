# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-004`

## Where we are

Repository: `robzer13/indice_nexus`  
Branch: `vnext`  
Gate 18: `IN_PROGRESS_NOT_FROZEN`

## v1.1 validation contract

The user explicitly authorized the versioned v1.1 validation-contract change.

Implementation:

`runtime/vnext/model-calibration-validation-v11.ts`

Contract:

`calibration/vnext/OROTITAN_GATE18_V1_1_VALIDATION_CONTRACT_001.json`

Authorization:

`calibration/vnext/OROTITAN_GATE18_V1_1_CONTRACT_CHANGE_AUTH_001.json`

Status: `IMPLEMENTED_CI_PASS`.

PR: #257.

## Verified architecture

v1.1 is additive. The v1.0 runtime, prompt, generation schema, and historical results are not rewritten.

Validation order:

1. raw schema;
2. raw presentation compliance;
3. safe presentation normalization on a copy;
4. presentation compliance re-check;
5. frozen v1.0 substantive semantic validator;
6. human-quality adjudication remains separate.

Safe normalization is restricted to appending one period while preserving all existing characters. It is forbidden when the normalized trimmed field would reach or exceed the frozen 178-character boundary.

If presentation remains noncompliant, substantive semantics are `NOT_EVALUATED_PRESENTATION_BLOCKER`, not semantic FAIL.

An invariant guard throws if a v1.0 presentation error leaks through after the v1.1 presentation layer.

## CI verification

- VNext lint: PASS
- VNext typecheck: PASS
- VNext unit/contract tests: PASS
- PostgreSQL migration tests: PASS
- production build: PASS
- Screener CI: PASS

## Preserved truths

- historical v1.0 statuses remain immutable;
- no retroactive PASS;
- Phi-4 STMicro raw v1.0 remains FAIL;
- Qwen3 4B Brookfield and RATIONAL substantive failures remain;
- no human-quality reassessment;
- no model winner;
- routing is not frozen.

## Still not authorized

- new model inference;
- second Phi-4 C4 cell;
- Qwen3.5 download;
- model switch;
- production mutation;
- publication.

## Exact next action

```text
DECIDE_PHI4_C4_RESUMPTION_UNDER_V1_1
```

The v1.1 contract is complete. The next decision is separate: whether to resume Phi-4 C4 qualification under this contract and, if so, which exact bounded cell is authorized.
