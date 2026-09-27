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


## D-2026-09-27-016 — Qwen3.5 forensic fails on private-artifact schema mismatch

Observed:
The read-only forensic passed the run identity checks but could not find a non-empty string at `$.response.rawText`, raising `VNEXT_GATE18_QWEN35_JSON_FORENSIC_RAW_TEXT_MISSING`.

Classification:
`FORENSIC_INPUT_SCHEMA_MISMATCH`.

Boundary:
This is a forensic-tooling defect. It does not change the Qwen3.5 model-run result, does not add evidence about model capability, and does not authorize a retry.

Remediation:
Add a read-only shape-discovery fallback that reports paths, types, string lengths, object-start flags, and SHA-256 hashes without printing raw values. Then rerun the same private artifact with no inference or Ollama call.


## D-2026-09-27-017 — Qwen3.5 forensic confirms empty final response, not partial JSON

Observed:
- `$.response.rawText` exists;
- type is string;
- length is 0;
- SHA-256 is the empty-string digest;
- `$.response.parsedJson` is null.

Conclusion:
There is no partial JSON body to structurally repair or semantically inspect.

Boundary:
The 842 generated tokens are not attributed to a thinking field as a proven fact because the original runner did not persist provider thinking output.

## D-2026-09-27-018 — First Qwen3.5 attempt is runtime-adapter confounded; authorize one think-false retry

External runtime evidence:
Ollama exposes explicit thinking-mode control for thinking-capable models, and Qwen3.5 is marked thinking-capable.

Decision:
Preserve the first run as FAIL, but do not treat it as a clean structured-output capability test because the runner omitted explicit thinking-mode control.

Authorize exactly one same-cell retry with `think:false`.

Frozen:
- same model/digest;
- same company;
- same packet/prompt/schema;
- context 16384;
- output 1024;
- temperature 0;
- timeout 600 seconds;
- validation v1.1.

No automatic second retry.

## D-2026-09-27-019 — Qwen3.5 think-false compatibility retry reaches engineering PASS

Observed execution:
- exactly one authorized same-cell local retry;
- model `qwen3.5:4b-q4_K_M` with pinned digest unchanged;
- Constellation Software / SERIAL_ACQUIRER;
- `think:false` as the sole runtime-compatibility change;
- wall clock 283571 ms;
- done reason `stop`;
- 953 / 1024 output tokens;
- 71 configured output tokens remained;
- no runtime error;
- schema PASS;
- raw presentation compliance PASS;
- v1.1 safe normalization count 0;
- substantive deterministic semantics PASS;
- zero external model API cost.

Disposition:
`ENGINEERING_PASS_HUMAN_ADJUDICATION_PENDING`.

Boundary:
The historical first attempt remains FAIL and is not reclassified. The retry authorization is consumed. No automatic second retry, broader Qwen3.5 expansion, model ranking, routing decision, production mutation, or publication authority follows from this engineering result.

Next action:
Human-quality adjudication of the private raw output against the pinned private evidence packet.

## D-2026-09-27-020 — Qwen3.5 Constellation human adjudication finds critical grounding failure

Observed:
- the clean `think:false` retry remains an engineering PASS;
- raw presentation compliance PASS with zero normalization;
- substantive deterministic semantics PASS;
- human adjudication finds exact-evidence-grounding failure.

Material defect:
`E-045` describes a replacement RFP after approximately 20 years of operation, while the generated finding states that the customer replaced the system.

Additional defects:
- exact duplicate weak-link candidates;
- low-information unresolved points;
- priority selection underweights serial-acquirer-specific moat evidence.

Disposition:
Stop Qwen3.5 C4 expansion and retain the candidate as calibration evidence.

Boundary:
No global Qwen3.5-family failure is inferred. No model ranking, routing freeze, production mutation, retry, or retroactive pass follows.

## D-2026-09-27-021 — Gemma 3 terms accepted and pinned download authorized

Public verification:
- exact Ollama tag `gemma3:4b-it-q4_K_M`;
- public digest prefix `a2af6cc3eb7f`;
- approximately 3.3GB;
- 128K context;
- Q4_K_M target.

Terms:
Gemma Terms of Use last modified 2026-04-01 were reviewed. The user explicitly stated `j’accepte les Gemma Terms`.

Static hardware disposition:
`PLAUSIBLE_BUT_TIGHT_UNPROVEN`.

Authorized action:
Exactly one local pinned download plus identity verification. No load smoke, prompt, inference, automatic retry, model switch, production mutation, or paid action.

Next:
After a successful pinned download and full digest capture, prepare a separate context4096 load-only memory preflight.

## D-2026-09-27-022 — Gemma 3 pinned download passes exact identity verification

Local execution result:
- model `gemma3:4b-it-q4_K_M`;
- full digest `a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`;
- size 3,338,801,804 bytes;
- format `gguf`;
- family `gemma3`;
- parameter size `4.3B`;
- quantization `Q4_K_M`.

Verification:
Exact tag, expected digest prefix, API-show reachability, and quantization all PASS.

Safety:
The download run performed no load smoke, prompt, semantic inference, retry, paid execution, production mutation, or publication.

## D-2026-09-27-023 — Gemma 3 context4096 load-only preflight authorized

Authority:
`OROTITAN-STANDING-TECHNICAL-AUTH-002`.

Scope:
One zero-cost load-only execution at context 4096 using the exact full digest captured above.

Guards:
- no prompt;
- no semantic inference;
- exact model identity required;
- explicit unload;
- no automatic retry;
- no context change;
- no model switch;
- no production mutation.

Next:
Execute the context4096 load-only preflight and use the measured RAM/VRAM/process split to decide whether an intermediate context8192 load-only step is justified.

## D-2026-09-27-024 — Gemma 3 context8192 load-only preflight passes

Observed:
- context 8192;
- VRAM used 2423 MiB;
- VRAM headroom 1540 MiB;
- free system RAM while loaded 0.77 GiB;
- processor split 56%/44% CPU/GPU;
- load-only guard PASS;
- explicit unload PASS;
- no prompt;
- no semantic inference.

Reference:
At context8192, Qwen3.5 used 2525 MiB VRAM with 1438 MiB headroom and 0.56 GiB loaded free RAM.

Interpretation:
Gemma 3 is not hardware-rejected at 8192 and has modestly better observed memory headroom than the Qwen3.5 reference. System RAM pressure remains high, so inference is still not authorized.

## D-2026-09-27-025 — Gemma 3 context16384 load-only preflight authorized

Authority:
`OROTITAN-STANDING-TECHNICAL-AUTH-002`.

Scope:
One zero-cost load-only execution at context 16384 using the exact pinned digest.

Guards:
- exact model identity required;
- no prompt;
- no semantic inference;
- explicit unload;
- no automatic retry;
- no context growth beyond 16384;
- no model switch;
- no production mutation.

Next:
If context16384 load fit passes, close the hardware qualification for the intended C4 target context and decide the first bounded Gemma 3 C4 discriminator separately.

## D-2026-09-27-026 — Gemma 3 context16384 load-only passes with high RAM pressure

Observed:
- context 16384;
- VRAM used 2449 MiB;
- VRAM headroom 1514 MiB;
- free system RAM while loaded 0.33 GiB;
- processor split 57%/43% CPU/GPU;
- load-only guard PASS;
- explicit unload PASS;
- no prompt;
- no semantic inference.

Reference:
At context16384, Qwen3.5 used 2589 MiB VRAM with 1374 MiB headroom and 0.51 GiB loaded free RAM.

Disposition:
Gemma 3 is hardware-admissible for the intended C4 context, but only with a HIGH_RAM_PRESSURE carry. Context growth beyond 16384 is not justified.

## D-2026-09-27-027 — Gemma 3 first C4 discriminator selects Constellation same packet

Selected:
`Constellation Software / SERIAL_ACQUIRER`.

Pinned invariants:
- packet SHA256 `9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8`;
- prompt SHA256 `0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8`;
- prompt bytes 9041;
- context 16384;
- max output 1024;
- temperature 0;
- timeout 600000 ms;
- validation contract v1.1.

Rationale:
The identical packet directly tests whether Gemma 3 improves the exact-grounding and priority-selection defects that caused the clean Qwen3.5 retry to fail human adjudication.

Authority:
One zero-cost local inference under `OROTITAN-STANDING-TECHNICAL-AUTH-002`.

Boundary:
No automatic retry, no second Gemma cell, no model ranking, no routing freeze, no production mutation, and no public generated content. Human adjudication is mandatory after an engineering PASS.

## D-2026-09-27-028 — Gemma 3 first Constellation C4 run fails deterministic substantive semantics

Observed:
- same pinned Constellation packet and prompt;
- context 16384;
- max output 1024;
- temperature 0;
- timeout 600000 ms;
- wall clock 142518 ms;
- done reason `stop`;
- prompt eval count 3558;
- eval count 683;
- output token margin 341;
- runtime error null;
- schema valid true;
- semantic valid false;
- semantic error `VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_WITHOUT_IDS`.

Validation v1.1:
- raw presentation compliant false;
- 13 presentation-only paths normalized;
- substantive validation remains FAIL.

Interpretation:
This is not runtime failure, schema failure, or output-budget exhaustion. The run remains historical FAIL and no retry is authorized.

## D-2026-09-27-029 — Read-only Gemma Constellation counterevidence forensic selected

Reason:
The first frozen semantic error means at least one priority finding contains a non-null `counterevidence_link` with an empty `counterevidence_ids` array.

Procedure:
- reproduce v1.1 presentation normalization in memory;
- identify violating finding indexes without printing private narrative text;
- set only the structurally invalid links to null on an in-memory copy;
- rerun the frozen v1.0 semantic validator;
- report any downstream deterministic error.

Boundary:
No inference, no Ollama call, no external network, no artifact mutation, no auto-repair, no retroactive pass, and no retry authority.

## D-2026-09-27-030 — Gemma Constellation first semantic defect affects all three findings

Read-only forensic result:
- v1.1 presentation normalization reproduced exactly with 13 normalized paths;
- finding indexes 1, 2, and 3 each had empty `counterevidence_ids` with a non-null `counterevidence_link`;
- the source artifact was not mutated;
- no inference was executed.

After setting only those structurally invalid links to null on an in-memory diagnostic copy, the frozen v1.0 semantic validator still failed.

Downstream error:

`VNEXT_GATE18_V10_UNKNOWN_CONFLICT_REF`

Conclusion:
The first deterministic semantic defect is not isolated. The historical Gemma run remains FAIL and no retry is authorized.

## D-2026-09-27-031 — Gemma Constellation unknown-conflict-reference forensic selected

Purpose:
Audit all conflict references by output section against the canonical packet conflict IDs, without printing private narrative text.

Diagnostic procedure:
- reproduce v1.1 presentation normalization;
- reproduce the first narrow counterevidence-link normalization;
- identify unknown conflict refs in priority findings, material conflicts, weak links, and unresolved points;
- remove only unknown conflict refs or unknown material-conflict entries on an in-memory copy;
- rerun the frozen v1.0 semantic validator.

Boundary:
No model call, no inference, no network, no artifact mutation, no auto-repair, no retroactive pass, and no retry authority.

## D-2026-09-27-032 — Gemma Constellation deterministic forensics exhausted

Second read-only forensic result:
- canonical conflict IDs: `C-005` only;
- model-generated unknown conflict ID: `C-006`;
- unknown `C-006` material-conflict entry count: 1;
- unknown `C-006` unresolved-point references: 2;
- total unknown conflict references removed on diagnostic copy: 3;
- cumulative diagnostic normalization also includes the three invalid counterevidence links from the first forensic;
- frozen v1.0 semantic validator after cumulative diagnostic normalization: PASS.

Conclusion:
Two deterministic substantive defect classes are established and no third known deterministic defect remains. The historical run remains FAIL. Engineering PASS and human-quality adjudication were not reached.

## D-2026-09-27-033 — Stop Gemma 3 C4 expansion

Disposition:

`STOP_GEMMA3_C4_EXPANSION_RETAIN_CALIBRATION_EVIDENCE`

Reason:
The first same-packet Constellation run failed engineering semantics with two independent substantive defect classes, including a fabricated conflict ID reused across multiple output sections. A same-prompt temperature-zero retry has low expected information gain relative to testing a new family.

Boundary:
No global Gemma-family failure is inferred. No winner is selected. Routing remains unfrozen.

## D-2026-09-27-034 — Refresh local candidate registry before Qwen3 8B

The registered `qwen3:8b` fallback remains on hold because its ~5.2GB artifact is a poor fit for the current 7.84 GiB RAM / 4 GiB VRAM envelope.

Refreshed candidates:
1. Granite 4 3B;
2. Ministral 3 3B Instruct;
3. Llama 3.2 3B;
4. Qwen3 8B on hardware-fit hold.

Selected next candidate:

`GRANITE4_3B_OLLAMA_Q4_K_M`

Selection basis:
new family diversity, ~2.1GB Q4_K_M artifact, Apache-2.0 license, 128K context, and strong instruction-following/tool-calling positioning.

## D-2026-09-27-035 — Granite 4 3B pinned download authorized

Authority:
`OROTITAN-STANDING-TECHNICAL-AUTH-002`.

Scope:
- exact tag `granite4:3b`;
- expected digest prefix `89962fcc7523`;
- expected quantization `Q4_K_M`;
- one download only;
- identity verification after download.

Not authorized in this step:
- load smoke;
- prompt;
- inference;
- automatic retry;
- automatic model switch;
- paid execution;
- production mutation.

## D-2026-09-27-036 — Granite 4 3B pinned download passes exact identity verification

Observed local identity:
- model `granite4:3b`;
- full digest `89962fcc75239ac434cdebceb6b7e0669397f92eaef9c487774b718bc36a3e5f`;
- size 2,099,521,385 bytes;
- format `gguf`;
- family `granite`;
- parameter size `3.4B`;
- quantization `Q4_K_M`.

No load or inference occurred.

## D-2026-09-27-037 — Granite 4 context4096 load-only preflight authorized

One zero-cost load-only execution is authorized at context 4096 using the exact full digest.

No prompt, semantic inference, retry, model switch, context change, or production mutation is authorized.

## D-2026-09-27-038 — Qwen3 8B explicitly queued immediately after Granite

User direction:
Test Qwen3 8B after Granite.

This overrides the prior informational ordering that placed Ministral 3 3B and Llama 3.2 3B before Qwen3 8B.

Required Qwen3 8B sequence:
1. pinned download and identity verification;
2. context4096 load-only;
3. context8192 load-only only if 4096 supports it;
4. context16384 load-only only if 8192 supports it;
5. inference only if the measured hardware qualification supports it.

Qwen3 8B remains recognized as a high-risk hardware candidate, but it is no longer deferred behind Ministral/Llama.

## D-2026-09-27-039 — Granite 4 context4096 load-only passes with strong relative headroom

Measured:
- free RAM before: 1.48 GiB;
- free RAM loaded: 1.06 GiB;
- free RAM after unload: 1.71 GiB;
- VRAM used loaded: 2313 MiB;
- VRAM free loaded: 1650 MiB;
- processor split: 15%/85% CPU/GPU;
- explicit unload: complete.

No prompt or inference occurred.

Relative to Gemma 3 at context4096:
- +0.48 GiB free RAM headroom;
- +136 MiB free VRAM headroom;
- materially higher GPU residency.

Conclusion:
Granite 4 context4096 hardware fit passes and supports one bounded context8192 load-only preflight.

## D-2026-09-27-040 — Granite 4 context8192 load-only authorized

Authority:
`OROTITAN-STANDING-TECHNICAL-AUTH-002`.

Scope:
- exact model `granite4:3b`;
- exact digest `89962fcc75239ac434cdebceb6b7e0669397f92eaef9c487774b718bc36a3e5f`;
- context 8192;
- one load-only execution;
- explicit unload.

Not authorized:
- prompt;
- semantic inference;
- context growth beyond 8192;
- automatic retry;
- model switch;
- production mutation.

Qwen3 8B remains the user-directed candidate immediately after Granite.

## D-2026-09-27-041 — Granite 4 context8192 load-only passes with useful headroom

Measured:
- free RAM before: 1.84 GiB;
- free RAM loaded: 0.94 GiB;
- free RAM after unload: 1.80 GiB;
- VRAM used loaded: 2355 MiB;
- VRAM free loaded: 1608 MiB;
- processor split: 22%/78% CPU/GPU;
- explicit unload: complete.

No prompt or inference occurred.

Relative to Granite 4096:
- loaded free RAM -0.12 GiB;
- VRAM headroom -42 MiB;
- GPU residency remains 78%.

Relative to Gemma 3 at context8192:
- +0.17 GiB loaded free RAM;
- +68 MiB VRAM headroom;
- materially higher GPU residency.

Conclusion:
Granite 4 context8192 hardware fit passes and supports one bounded context16384 load-only preflight.

## D-2026-09-27-042 — Granite 4 context16384 load-only authorized

Authority:
`OROTITAN-STANDING-TECHNICAL-AUTH-002`.

Scope:
- exact model `granite4:3b`;
- exact digest `89962fcc75239ac434cdebceb6b7e0669397f92eaef9c487774b718bc36a3e5f`;
- context 16384;
- one load-only execution;
- explicit unload.

Not authorized:
- prompt;
- semantic inference;
- context growth beyond 16384;
- automatic retry;
- model switch;
- production mutation.

Qwen3 8B remains the user-directed candidate immediately after Granite.

## D-2026-09-27-043 — Granite 4 context16384 load-only passes with high RAM pressure

Measured:
- free RAM before: 1.30 GiB;
- free RAM loaded: 0.41 GiB;
- free RAM after unload: 1.82 GiB;
- VRAM used loaded: 2305 MiB;
- VRAM free loaded: 1658 MiB;
- processor split: 38%/62% CPU/GPU;
- explicit unload: complete.

No prompt or inference occurred.

Relative to Granite 8192:
- loaded free RAM -0.53 GiB;
- VRAM headroom +50 MiB due to greater CPU offload;
- GPU residency decreases from 78% to 62%.

Relative to Gemma 3 at context16384:
- +0.08 GiB loaded free RAM;
- +144 MiB VRAM headroom;
- higher GPU residency.

Disposition:
`PASS_WITH_HIGH_RAM_PRESSURE`.

No context growth beyond 16384 is authorized.

## D-2026-09-27-044 — Granite 4 first C4 discriminator selects Constellation same packet

Selected:
`Constellation Software / SERIAL_ACQUIRER`.

Pinned:
- packet SHA256 `9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8`;
- prompt SHA256 `0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8`;
- context 16384;
- max output 1024;
- temperature 0;
- timeout 600000 ms;
- validation v1.1.

Authority:
One zero-cost local inference under `OROTITAN-STANDING-TECHNICAL-AUTH-002`.

Boundary:
No automatic retry, no additional Granite C4 cell, no model ranking, no routing freeze, and no production mutation before the result is evaluated. Qwen3 8B remains next after Granite.

## D-2026-09-27-045 — Granite 4 Constellation reaches engineering PASS

Observed:
- schema valid;
- v1.1 substantive semantics PASS;
- five missing-terminal-punctuation issues safely normalized;
- wall clock 117903 ms;
- eval count 536;
- runtime error null.

Engineering disposition:
`ENGINEERING_PASS_WITH_PRESENTATION_COMPLIANCE_CARRY`.

## D-2026-09-27-046 — Granite 4 human adjudication is critical failure

Human defects:
- E-029 overgeneralizes acquisition criteria into verified properties of every acquired company;
- the universal E-029 claim has no qualification;
- two of three priority findings are near-duplicate recurring-revenue share metrics;
- unresolved retention question reuses E-063 although prior same-packet adjudication established the disclosure is already answered;
- E-066/E-067 direct serial-acquirer moat inputs are omitted.

Positive:
- no evidence-ID invention;
- no conflict-ID invention;
- C-005 is surfaced;
- the recurring-revenue-vs-retention weak link is directionally useful;
- no final moat judgment emitted.

Disposition:
`STOP_GRANITE4_C4_EXPANSION_RETAIN_CALIBRATION_EVIDENCE`.

No global Granite-family failure inferred.

## D-2026-09-27-047 — Qwen3 8B user-directed sequence becomes active

Public Ollama target:
`qwen3:8b-q4_K_M`

Verified public metadata:
- digest prefix `500a1f067a9f`;
- ~5.2GB;
- 8.19B parameters;
- Q4_K_M;
- 40K context;
- Apache-2.0.

Exactly one zero-cost pinned download is authorized under `OROTITAN-STANDING-TECHNICAL-AUTH-002`.

No load smoke, inference, retry, model switch, paid execution, or production mutation is authorized in the download step.

## D-2026-09-27-048 — Qwen3 8B pinned download passes exact identity verification

Observed local identity:
- model `qwen3:8b-q4_K_M`;
- full digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`;
- size 5,225,388,164 bytes;
- format `gguf`;
- family `qwen3`;
- parameter size `8.2B`;
- quantization `Q4_K_M`.

No load or inference occurred.

## D-2026-09-27-049 — Qwen3 8B context4096 load-only preflight authorized

Authority:
`OROTITAN-STANDING-TECHNICAL-AUTH-002`.

Scope:
- exact pinned model digest;
- context 4096;
- one load-only execution;
- explicit unload.

Hardware risk:
`HIGH_MEMORY_PRESSURE_EXPECTED`.

Not authorized:
- prompt;
- semantic inference;
- context growth;
- automatic retry;
- model switch;
- production mutation.

Any context8192 step requires review of the measured 4096 result first.

## D-2026-09-27-050 — Qwen3 8B context4096 load-only passes with critical RAM pressure

Measured:
- free RAM before: 1.38 GiB;
- free RAM loaded: 0.21 GiB;
- free RAM after unload: 3.75 GiB;
- VRAM used loaded: 2279 MiB;
- VRAM free loaded: 1684 MiB;
- processor split: 61%/39% CPU/GPU;
- Ollama resident size: 6.0 GB;
- explicit unload: complete.

No prompt or inference occurred.

Disposition:
`PASS_WITH_CRITICAL_RAM_PRESSURE`.

The result proves loadability only, not inference fit or production fit.

## D-2026-09-27-051 — Qwen3 8B context8192 diagnostic load-only authorized

A single context8192 load-only diagnostic is authorized under `OROTITAN-STANDING-TECHNICAL-AUTH-002`.

Reason:
- context4096 loaded successfully;
- target C4 prompt plus output envelope is not comfortably represented by a 4096 context budget;
- measured VRAM headroom remains substantial;
- empirical 8192 loadability is more informative than extrapolation alone.

Boundary:
- diagnostic-only;
- no prompt;
- no inference;
- no context16384 pre-authorization;
- no automatic retry;
- no model switch;
- production mutation forbidden.

## D-2026-09-27-052 — Qwen3 8B context8192 load-only passes with critical RAM pressure

Measured:
- free RAM before: 1.20 GiB;
- free RAM loaded: 0.22 GiB;
- free RAM after unload: 3.07 GiB;
- VRAM used loaded: 2323 MiB;
- VRAM free loaded: 1640 MiB;
- processor split: 64%/36% CPU/GPU;
- resident size: 6.6 GB;
- explicit unload: complete.

No prompt or inference occurred.

Disposition:
`PASS_WITH_CRITICAL_RAM_PRESSURE`.

## D-2026-09-27-053 — Qwen3 8B context16384 final load-only diagnostic authorized

Reason:
- context8192 loadability is confirmed;
- memory pressure is severe but stable relative to context4096;
- exact Constellation comparability uses context16384;
- a load-only measurement has high information value and zero external monetary cost.

Boundary:
- exactly one context16384 load-only execution;
- no prompt;
- no inference;
- no further context growth;
- no automatic retry/model switch;
- explicit post-run decision required before any C4 inference.

## D-2026-09-27-054 — Qwen3 8B context16384 load-only passes with extreme RAM pressure

Measured:
- free RAM before: 2.16 GiB;
- free RAM loaded: 0.14 GiB;
- free RAM after unload: 1.31 GiB;
- VRAM used loaded: 2345 MiB;
- VRAM free loaded: 1618 MiB;
- processor split: 70%/30% CPU/GPU;
- resident size: 7.8 GB;
- explicit unload: complete.

No prompt or inference occurred.

Disposition:
`PASS_WITH_EXTREME_RAM_PRESSURE`.

No further context growth is authorized. Production fit is not established.

## D-2026-09-27-055 — One bounded Qwen3 8B Constellation think-false experiment authorized

Reason:
- the user explicitly required Qwen3 8B to be tested after Granite;
- Qwen3 8B completed staged load-only qualification through context16384;
- the exact comparable Constellation packet fits the selected context;
- Ollama supports explicit Qwen3 thinking disablement;
- `think:false` follows the already-established Qwen-family structured-output compatibility control.

Pinned:
- exact model digest;
- Constellation packet/prompt hashes unchanged;
- context 16384;
- max output 1024;
- temperature 0;
- timeout 600000 ms;
- `think:false`;
- validation v1.1.

Safety/diagnostic guard:
- minimum 1.0 GiB free system RAM before inference begins;
- no other loaded Ollama model;
- one run only;
- no retry, context change, model switch, or production mutation.

The experiment can establish a model result for this cell; it cannot by itself establish production hardware suitability.

## D-2026-09-27-056 — Qwen3 8B Constellation C4 reaches clean engineering PASS

Observed:
- baseline free RAM 3.59 GiB, above 1.0 GiB guard;
- wall clock 519551 ms;
- done reason stop;
- prompt eval 3356;
- eval 860 / 1024;
- runtime error null;
- schema valid;
- raw presentation compliant;
- zero normalization paths;
- substantive semantics PASS.

Disposition:
`ENGINEERING_PASS_HUMAN_ADJUDICATION_REQUIRED`.

The result does not establish human-quality PASS, model ranking, routing, or production hardware suitability.

## D-2026-09-27-057 — Qwen3 8B human adjudication requires private bundle review

The C4 authorization is consumed. No retry or additional inference is authorized.

A private local bundle builder is prepared to combine:
- the exact persisted model response;
- the exact pinned Constellation evidence packet;
- validator telemetry;
- known same-packet human-regression checks.

The bundle remains under `calibration/vnext/private-runs/` and must not be committed or published.

## D-2026-09-27-058 — Qwen3 8B clean engineering PASS does not survive human adjudication

Human adjudication of the exact private Constellation packet/output finds a critical quality failure.

Observed defects:
- E-045 replacement RFP/process is overstated as eventual replacement;
- E-063 is re-opened as a retention/churn question despite the packet already defining nondisclosure as not evidence of low retention;
- priority selection omits stronger direct evidence, including E-043 and E-067;
- E-066 is surfaced only as an unresolved point rather than used as a direct serial-acquirer moat input.

Positive controls:
- no evidence-ID invention;
- no conflict-ID invention;
- C-005 remains surfaced and unresolved;
- no final moat judgment is emitted;
- engineering validation remains a clean PASS with zero normalization.

Disposition:
`STOP_QWEN3_8B_C4_EXPANSION_RETAIN_CALIBRATION_EVIDENCE`.

This is not a global Qwen3-family failure conclusion.

## D-2026-09-27-059 — Ministral 3 3B becomes the next local candidate

The refreshed candidate registry selects `ministral-3:3b-instruct-2512-q4_K_M` after completion of the explicit Granite -> Qwen3 8B sequence.

Pinned public identity:
- digest prefix f04aa1c738f6;
- ~3.0 GB artifact;
- 3.85B parameters;
- Q4_K_M;
- 256K context;
- Apache-2.0.

Exactly one zero-cost pinned download and identity verification is authorized under `OROTITAN-STANDING-TECHNICAL-AUTH-002`. Load and inference remain unauthorized.

## D-2026-09-27-060 — Ministral 3 3B pinned download passes exact local identity verification

Observed:
- exact tag `ministral-3:3b-instruct-2512-q4_K_M`;
- full digest `f04aa1c738f64e13c625b82ae92504fc0260fa6723b509ed1ece0fa188179b1d`;
- size 2,953,840,808 bytes;
- format `gguf`;
- family `mistral3`;
- Ollama parameter size `3.8B`;
- quantization `Q4_K_M`.

No load or inference occurred.

## D-2026-09-27-061 — Ministral 3 3B context4096 load-only preflight authorized

Exactly one context4096 load-only run is authorized under `OROTITAN-STANDING-TECHNICAL-AUTH-002`.

No prompt, inference, retry, context growth, model switch, paid execution, or production mutation is authorized.

## D-2026-09-27-062 — Ministral 3 3B context4096 load-only passes with critical RAM pressure

Measured:
- free RAM before: 1.14 GiB;
- free RAM loaded: 0.35 GiB;
- free RAM after unload: 1.85 GiB;
- VRAM used loaded: 2397 MiB;
- VRAM free loaded: 1566 MiB;
- processor split: 48%/52% CPU/GPU;
- resident size: 3.1 GB;
- explicit unload: complete.

No prompt or inference occurred.

Disposition:
`PASS_WITH_CRITICAL_RAM_PRESSURE`.

The result proves loadability only, not inference fit or production fit.

## D-2026-09-27-063 — Ministral 3 3B context8192 diagnostic load-only authorized

A single context8192 load-only diagnostic is authorized under `OROTITAN-STANDING-TECHNICAL-AUTH-002`.

Reason:
- context4096 loaded successfully;
- target C4 prompt plus output envelope is not comfortably represented by a 4096 context budget;
- the measured state is memory-constrained but remains comparable to prior candidates that proceeded through bounded staged qualification;
- empirical 8192 loadability is more informative than extrapolation alone.

Boundary:
- diagnostic-only;
- no prompt;
- no inference;
- no context16384 pre-authorization;
- no automatic retry;
- no model switch;
- production mutation forbidden.

## D-2026-09-27-064 — Ministral 3 3B context8192 load-only passes with critical RAM pressure

Measured:
- free RAM before: 1.00 GiB;
- free RAM loaded: 0.38 GiB;
- free RAM after unload: 2.06 GiB;
- VRAM used loaded: 2407 MiB;
- VRAM free loaded: 1556 MiB;
- processor split: 54%/46% CPU/GPU;
- resident size: 3.5 GB;
- explicit unload: complete.

No prompt or inference occurred.

Disposition:
`PASS_WITH_CRITICAL_RAM_PRESSURE`.

Relative to context4096, memory pressure is severe but stable enough to justify one final load-only measurement at context16384.

## D-2026-09-27-065 — Ministral 3 3B context16384 final load-only diagnostic authorized

Reason:
- context8192 loadability is confirmed;
- memory pressure is critical but does not materially degrade from context4096 to context8192;
- exact C4 comparability uses context16384;
- one final load-only measurement has high information value and zero external monetary cost.

Boundary:
- exactly one context16384 load-only execution;
- no prompt;
- no inference;
- no further context growth;
- no automatic retry/model switch;
- explicit post-run decision required before any C4 inference.

## D-2026-09-27-066 — Ministral 3 3B context16384 load-only passes

Measured:
- free RAM before: 1.45 GiB;
- free RAM loaded: 1.21 GiB;
- free RAM after unload: 3.88 GiB;
- VRAM used loaded: 2399 MiB;
- VRAM free loaded: 1564 MiB;
- processor split: 63%/37% CPU/GPU;
- resident size: 4.4 GB;
- explicit unload: complete.

No prompt or inference occurred.

Disposition:
`PASS_WITH_USEFUL_HEADROOM`.

Caveat:
free system RAM is host-state-sensitive across separate runs. The 1.21 GiB observation supports bounded execution but is not an intrinsic model memory-footprint estimate.

No context growth beyond 16384 is authorized.

## D-2026-09-27-067 — One bounded Ministral Constellation C4 inference authorized

The same pinned Constellation Software packet is selected for direct protocol comparability.

Execution envelope:
- context 16384;
- max output 1024;
- temperature 0;
- timeout 600000 ms;
- keep-alive 0s;
- loopback transport only;
- minimum baseline free RAM 1.0 GiB;
- no other Ollama model loaded.

Boundary:
- exactly one inference;
- no automatic retry;
- no prompt/schema/packet/context/output/temperature/timeout/model change;
- raw generated content remains private;
- human adjudication is required if engineering validation passes;
- no production, ranking or routing authority.

