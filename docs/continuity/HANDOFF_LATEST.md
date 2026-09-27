# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-007`

## Current state

Gate 18 remains `IN_PROGRESS_NOT_FROZEN`.

The authorized single Phi-4 / Constellation Software C4 run has been executed exactly once and the authorization is consumed.

## Engineering result

Result: `ENGINEERING_PASS_HUMAN_ADJUDICATION_PENDING`

- runtime: PASS
- wall clock: 209989 ms
- done reason: `stop`
- prompt eval count: 3169
- output eval count: 681 / 1024
- output token margin: 343
- runtime error: none
- schema: PASS
- raw presentation compliance: FAIL
- v1.1 normalized paths: 3
- v1.1 substantive deterministic semantics: PASS

Raw private output:

`calibration/vnext/private-runs/2026-09-27T085515615Z__C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_PHI4MINI_V1_1_CONTEXT16384_001.json`

The raw private content is not persisted publicly.

## Interpretation boundary

The run demonstrates that Phi-4 can produce a schema-valid, substantively valid Constellation output under v1.1 after three safe presentation normalizations.

It does not yet complete the matrix cell because human-quality adjudication is mandatory.

The recurring presentation-compliance defect remains real and must be carried separately.

## Authorization state

The single-run authorization is consumed.

No retry, second attempt, new inference, model switch, production mutation, or publication is authorized.

## Exact next action

```text
REVIEW_PRIVATE_RAW_OUTPUT_AND_COMPLETE_HUMAN_ADJUDICATION
```
