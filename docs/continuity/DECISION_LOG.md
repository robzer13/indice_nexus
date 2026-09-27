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


## D-2026-09-27-010 — Authorized Phi-4 Constellation v1.1 run consumed with engineering PASS

Observed execution:
- exactly one authorized local run;
- done reason `stop`;
- 681 / 1024 output tokens;
- no runtime error;
- schema PASS;
- raw presentation noncompliant;
- 3 safe v1.1 normalization paths;
- substantive deterministic semantics PASS.

Disposition:
`ENGINEERING_PASS_HUMAN_ADJUDICATION_PENDING`.

Boundary:
The authorization is consumed. No retry is authorized. The matrix cell is not complete until the private raw output receives human-quality adjudication.


## D-2026-09-27-011 — Phi-4 expansion stopped after Constellation human critical failure

Decision:
Stop further Phi-4 C4 expansion and retain the model as calibration evidence.

Reason:
The v1.1 architecture resolved the presentation-layer confound, but the Constellation cell exposed a separate critical human-quality failure in evidence grounding, causal usefulness, and priority selection.

Boundary:
No global Phi-4 family failure is inferred from this single post-v1.1 cell.

## D-2026-09-27-012 — Standing zero-cost execution authorization activated

Authority:
The user instructed that OroTitan zero-cost technical execution should proceed without repeated confirmation.

Operational rule:
- zero-cost execution: proceed under `OROTITAN-STANDING-TECHNICAL-AUTH-002`;
- nonzero external monetary cost: obtain explicit user approval before spending.

Protocol guards remain unchanged.

## D-2026-09-27-013 — Qwen3.5 4B target-context hardware path passes load-only qualification

Observed:
- context 4096 PASS;
- context 8192 PASS;
- context 16384 PASS;
- exact model identity and digest pinned;
- no semantic inference during load preflights;
- full unload confirmed after each measurement.

At context 16384:
- VRAM used: 2589 MiB;
- VRAM headroom: 1374 MiB;
- Ollama processor split: 56% CPU / 44% GPU;
- loaded free RAM observed: 0.51 GiB.

Interpretation:
Target context is loadable, but RAM pressure and CPU offload are material.

## D-2026-09-27-014 — Select Constellation as first bounded Qwen3.5 C4 discriminator

Decision:
Use Constellation Software / SERIAL_ACQUIRER as the first Qwen3.5 C4 cell.

Rationale:
The same pinned packet already separates prior candidates:
- Qwen3 4B: engineering PASS / human PASS_WITH_CARRY;
- Phi-4: engineering PASS under v1.1 / human CRITICAL_FAILURE.

This makes Constellation the highest-information first discriminator for Qwen3.5 grounding and priority-selection quality.

Frozen execution:
- context 16384;
- max output 1024;
- temperature 0;
- timeout 600 seconds;
- generation prompt/schema v1.0 unchanged;
- validation v1.1;
- one run;
- no automatic retry;
- zero external model API cost.


## D-2026-09-27-015 — First Qwen3.5 C4 Constellation run fails raw JSON completeness

Observed execution:
- model `qwen3.5:4b-q4_K_M`;
- Constellation Software full pinned packet;
- context 16384;
- max output 1024;
- temperature 0;
- timeout 600 seconds;
- wall clock 249631 ms;
- done reason `stop`;
- eval count 842;
- no runtime error;
- JSON parse error `Unexpected end of JSON input`.

Interpretation:
This is not classified as a timeout and output-budget exhaustion is not proven because generation stopped with 182 tokens of configured output headroom remaining.

Disposition:
Preserve the run as FAIL. Do not reach v1.1 deterministic semantic or human-quality conclusions from malformed JSON. Run a read-only structural forensic on the existing private output before deciding whether a bounded reliability retry is informative.

Authority:
The inference authorization is consumed. The forensic is zero-cost and authorized under `OROTITAN-STANDING-TECHNICAL-AUTH-002`.
