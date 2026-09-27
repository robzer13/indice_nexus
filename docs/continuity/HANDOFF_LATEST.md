# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-003`

## Where we are

Repository: `robzer13/indice_nexus`  
Branch: `vnext`  
Last verified HEAD before this implementation checkpoint: `f70a0ca9533301f8f4e689bb596b5ff2a8ff1d31`

Gate 18 remains `IN_PROGRESS_NOT_FROZEN`.

## Authorization

The user explicitly authorized the versioned v1.1 validation-contract change on 2026-09-27.

Authorization artifact:

`calibration/vnext/OROTITAN_GATE18_V1_1_CONTRACT_CHANGE_AUTH_001.json`

This authorization covers the additive v1.1 validation architecture only. It does not authorize new inference, a second Phi-4 C4 cell, model switching, production mutation, publication, or historical-result mutation.

## v1.1 implementation

Runtime:

`runtime/vnext/model-calibration-validation-v11.ts`

Contract artifact:

`calibration/vnext/OROTITAN_GATE18_V1_1_VALIDATION_CONTRACT_001.json`

Architecture:

1. raw output schema validation;
2. raw presentation-compliance validation;
3. strict safe presentation normalization on a copy;
4. substantive deterministic semantic validation on the normalized copy;
5. human-quality adjudication remains separate.

The v1.0 prompt and generation schema are unchanged.

The safe normalizer may only append one period, preserve every existing character, never mutate the raw output, and only operate when the resulting trimmed length remains strictly below 178 characters.

If presentation remains blocked, substantive semantics are `NOT_EVALUATED_PRESENTATION_BLOCKER`, not a semantic FAIL.

The substantive layer reuses the frozen v1.0 semantic validators only after the normalized copy is fully presentation-compliant.

## Preserved truths

- historical v1.0 results remain immutable;
- no retroactive PASS;
- Phi-4 STMicro raw v1.0 remains FAIL;
- Qwen3 4B Brookfield and RATIONAL substantive failures remain;
- no human-quality reassessment;
- no model winner;
- routing not frozen.

## Current state

Implementation files and contract tests have been added on the implementation branch.

CI has not yet been accepted as the implementation close condition.

## Exact next action

```text
VERIFY_V1_1_CONTRACT_IMPLEMENTATION_CI
```

After CI passes, mark the versioned v1.1 validation contract implemented and move to the separate decision on whether Phi-4 C4 qualification should resume under v1.1.
