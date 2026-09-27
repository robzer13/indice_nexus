# Session Checkpoint — 2026-09-27 — Shadow Replay Complete

Resume ID: `VNEXT-G18-C-20260927-002`

## Input received

The user supplied the complete local contents of:

`calibration/vnext/private-runs/OROTITAN_GATE18_V1_1_SHADOW_REPLAY_RESULT_001.json`

Status: `SHADOW_REPLAY_COMPLETE`.

## Material findings

- 6 cases replayed.
- 5 raw outputs were presentation-noncompliant.
- 3 historical raw FAIL cases become deterministic semantic PASS after terminal-punctuation-only normalization on a shadow copy.
- Qwen3 4B Brookfield remains a substantive semantic FAIL.
- Qwen3 4B RATIONAL remains a substantive semantic FAIL and additionally contains one saturation-boundary path that is not safely normalized.
- Qwen3 4B STMicro positive control remains PASS.
- No new model inference was executed.
- No source artifact was mutated.
- No historical v1.0 status was changed.
- Human quality was not reassessed.

## Source-code validation

The v1.0 implementation was inspected.

`assertCompleteNarrative`:
- emits the field-specific supplied error code when terminal punctuation is missing;
- emits `VNEXT_GATE18_V10_NARRATIVE_BOUNDARY_SATURATION` separately when trimmed length is >= 178.

The targeted Adyen validator calls the same full v1.0 semantic validator.

This validates the replay taxonomy for the observed `*_INCOMPLETE` errors.

## Diagnostic disposition

`A_TWO_LAYER_VALIDATION_WITH_SAFE_NORMALIZATION` is supported by this six-case shadow replay as a contract-architecture hypothesis.

It is not implemented and implementation is not authorized.

## Next action

`DECIDE_AND_AUTHORIZE_OR_REJECT_VERSIONED_V1_1_CONTRACT_CHANGE`
