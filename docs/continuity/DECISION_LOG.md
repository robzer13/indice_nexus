# OroTitan VNExT — Decision Log

Append-only continuity log. This file records decisions already made through authorized OroTitan execution. It does not create analytical authority.

## D-2026-09-27-001 — Strategy checkpoint selects contract-architecture review

Decision:
Do not spend a second Phi-4 C4 matrix cell after recurrent narrative-completeness defects. Use a zero-inference discriminator first.

Preserved boundary:
No automatic model switch, contract change, candidate rejection, production mutation, or retroactive pass.

## D-2026-09-27-002 — Preferred review hypothesis is two-layer validation

Hypothesis:
Separate raw schema/presentation compliance from safe presentation normalization, substantive deterministic semantic validation, and human-quality adjudication.

Status:
Diagnostic hypothesis only. Not implemented as v1.1 contract.

## D-2026-09-27-003 — Shadow replay authorized under standing technical authority

Scope:
Replay six existing private artifacts with no new inference and no source mutation.

Normalization:
Terminal punctuation only, on a shadow copy, below the frozen safe narrative boundary.

## D-2026-09-27-004 — Replay failure-layer taxonomy corrected

Observation:
A changed downstream validator error is not automatically a substantive semantic failure. Errors such as `*_INCOMPLETE` and `NARRATIVE_BOUNDARY_SATURATION` can remain in presentation/compliance layers.

Action:
PR #254 introduced explicit diagnostic layers:

- `PRESENTATION_COMPLIANCE`
- `PRESENTATION_BOUNDARY`
- `SUBSTANTIVE_SEMANTIC`
- `SCHEMA`
- `UNKNOWN`

Tests:
Screener CI PASS.
VNext CI PASS.

Merge:
`223ba0052d5ad2b6b5e14b64290347463df904f0`

Methodology change:
NO.


## D-2026-09-27-005 — Shadow replay completes and supports two-layer architecture hypothesis

Evidence:
Six existing private artifacts were replayed with no inference and no source mutation.

Observed:
- three raw FAIL cases become deterministic semantic PASS after terminal-punctuation-only normalization;
- two Qwen3 4B full cases retain substantive semantic failures;
- one Qwen3 4B STMicro positive control remains PASS without normalization.

Source-code verification:
The relevant v1.0 `*_INCOMPLETE` errors are emitted by `assertCompleteNarrative` when terminal punctuation is missing. Narrative saturation has a separate error code.

Diagnostic conclusion:
The replay supports `A_TWO_LAYER_VALIDATION_WITH_SAFE_NORMALIZATION` as the preferred contract-architecture hypothesis for separating raw presentation compliance from substantive deterministic semantics.

Authority boundary:
This is not a v1.1 contract implementation or authorization. Historical v1.0 results remain immutable; human quality and model winner/routing are unchanged.
