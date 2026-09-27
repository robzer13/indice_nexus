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


## D-2026-09-27-006 — User authorizes versioned v1.1 validation contract change

Authority:
Explicit user authorization in chat: `ok autorisation`.

Authorized implementation:
`GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1`.

Normative boundary:
- keep v1.0 prompt and generation schema unchanged;
- preserve raw output and raw presentation-compliance result;
- safe normalization may append one period only;
- preserve all existing characters;
- no lexical repair, deletion, replacement, or reordering;
- never normalize into the >=178 saturation boundary;
- do not run substantive semantic validation while a presentation blocker remains;
- reuse frozen v1.0 semantic validators only after presentation compliance;
- keep human-quality adjudication separate;
- preserve all historical v1.0 results.

Not authorized:
New inference, second Phi-4 C4 cell, Qwen3.5 download, model switch, production mutation, publication, or retroactive pass.


## D-2026-09-27-007 — Versioned v1.1 validation contract implemented and CI-verified

Result:
`GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1` is implemented as an additive validation layer.

Verification:
- VNext CI PASS;
- Screener CI PASS;
- lint PASS;
- typecheck PASS;
- unit and contract tests PASS;
- PostgreSQL migration tests PASS;
- production build PASS.

Preserved boundary:
v1.0 remains unchanged and historical v1.0 outcomes are not reclassified.

Execution boundary:
No model inference was executed. A second Phi-4 C4 cell remains unauthorized.

Next decision:
`DECIDE_PHI4_C4_RESUMPTION_UNDER_V1_1`.


## D-2026-09-27-008 — Resume Phi-4 C4 with one Constellation discriminator under v1.1

Decision:
Resume qualification is methodologically justified after the v1.1 architecture fix, but only through one bounded discriminator before any broader matrix continuation.

Selected cell:
Constellation Software / SERIAL_ACQUIRER.

Rationale:
It is the smallest remaining workload, closest to the STMicro reference cell, adds a distinct archetype, is absent from the shadow replay, and had a Qwen3 4B engineering PASS / human PASS_WITH_CARRY reference result.

Frozen execution proposal:
16384 context, 1024 output tokens, temperature 0, 480-second client timeout, loopback transport, one run, no automatic retry.

Authority boundary:
The decision and runner preparation are authorized. The model inference itself is not yet authorized.


## D-2026-09-27-009 — User authorizes exactly one Phi-4 Constellation v1.1 inference

Authority:
Explicit user authorization in chat: `autorisé`.

Authorized execution:
Exactly one local `phi4-mini:3.8b-q4_K_M` C4 inference on Constellation Software using the frozen v1.1 cell.

Frozen parameters:
16384 context, 1024 max output tokens, temperature 0, 480-second client timeout, loopback transport, keep_alive 0s.

Execution boundary:
One run maximum. No automatic retry. No parameter change. No model switch. No production mutation. No publication. Historical v1.0 results remain immutable.
