# OROTITAN_GATE18_PHASE_C_PROTOCOL_V0.1

**Gate:** 18  
**Phase:** C — Diversified Local-First Model Qualification  
**Status:** DRAFT EXECUTION PROTOCOL — NOT FROZEN  
**Precondition:** Phase A = PASS, Phase B = PASS  
**Gate 18 status:** IN_PROGRESS_NOT_FROZEN  
**Production mutation:** FORBIDDEN  
**Publication authority:** DISABLED  
**Model winner:** NONE  
**Routing freeze:** NONE  
**Paid execution authorized by this protocol:** NO

## 1. Phase C objective

Phase C determines whether OroTitan can operate its analytical model layer with:

```text
TARGET PRODUCTION ECONOMICS = ZERO PER-CALL MODEL API COST
```

The preferred production candidate is therefore a model whose weights can be run on infrastructure controlled by the user without a metered external model API charge.

This is a **local/open-weight-first** qualification phase.

The objective is not to find the most capable model in the abstract.

The objective is:

> Find the lowest-resource local/open-weight model that satisfies OroTitan's analytical quality, evidence-role, structured-output and judgment-boundary requirements with acceptable reliability.

## 2. Meaning of "free"

For Gate 18 Phase C:

```text
FREE = ZERO PER-CALL EXTERNAL MODEL API CHARGE
```

This does not mean zero total cost of ownership.

Local execution may consume electricity, storage, RAM / VRAM, CPU / GPU time, download bandwidth and hardware already owned by the user.

These costs must be measured separately from API cost.

Phase C must never describe local inference as literally costless.

## 3. Model classes

### 3.1 Production candidates

Production candidates must be open-weight or otherwise locally deployable, runnable on user-controlled hardware, usable without a metered external inference API, compatible with deterministic prompt/schema pinning, capable of machine-parseable structured output, and identified by exact model + quantization + runtime version.

### 3.2 Paid models

Paid models such as LUNA / TERRA / SOL / ASTRA are:

```text
BENCHMARK_ONLY
```

They may be used only when a specific Phase C information gap justifies a controlled benchmark and only after no-cost dry-run, fresh budget snapshot, explicit single-call authorization, hard spend cap, and no automatic escalation.

Paid models are not production candidates for the zero-per-call target.

## 4. Phase C stages

### C0 — hardware and runtime qualification

No model execution.

Collect operating system, CPU identity and logical core count, system RAM, available NVIDIA GPU identity / VRAM when present, Windows display-adapter metadata as fallback, local runtime availability, Docker availability and Python availability.

C0 must not download a model, make an inference request, contact Vercel AI Gateway, mutate production, or write private research data.

C0 output is a local-machine capability receipt.

### C1 — local runtime admission

Admit one local inference runtime.

Preferred interface:

```text
LOCAL OPENAI-COMPATIBLE HTTP ENDPOINT
```

Reason: provider-neutral OroTitan adapter, runtime can change without rewriting analytical contracts, and model identity remains separate from transport.

Runtime admission must verify endpoint is loopback or explicitly user-controlled, model identifier is exact, no hidden cloud fallback exists, model is already downloaded locally before inference qualification, response model identity is traceable, and deterministic generation settings are recorded.

### C2 — structured-output smoke

Run a very small non-private synthetic prompt.

Pass requirements: schema parse succeeds, required IDs remain exact, no malformed evidence references, finish condition is normal, no retry is hidden, and latency/token counts are captured when exposed.

C2 is not a quality benchmark.

### C3 — historical semantic regression suite

First local quality tests must replay the exact failures Phase B discovered.

Minimum mandatory regressions:

```text
Adyen same-target switching-friction role alignment
Adyen scope-resolved incident handling
Adyen qualification orthogonality
Brookfield E-039 peer-platform replicability polarity
Brookfield E-042 competing-capital polarity
```

A production candidate that fails a previously closed critical semantic regression is not admitted to broader Phase C qualification.

### C4 — diversified company × module matrix

After C3 PASS, evaluate local candidates on a diversified matrix.

The matrix must span the five current company archetypes: SIMPLE_CLEAN_COMPOUNDER, SERIAL_ACQUIRER, CYCLICAL_SEGMENTED, ACCOUNTING_HEAVY, and DIFFICULT_MOAT_CONFLICTING_EVIDENCE.

Initial pinned companies may reuse RATIONAL AG, Constellation Software, STMicroelectronics, Brookfield Corporation, and Adyen.

Phase C may add new companies only through a separate corpus-expansion artifact.

Phase C must not confuse deterministic Gate 15 modules with model-dependent judgment tasks.

The frozen deterministic P0 module suite remains provider-independent:

- WEAK_LINK_TAXONOMY
- DECISION_STATE_ARCHITECTURE
- RETURN_NORMALIZATION
- CAPITAL_SEASONING
- OWNER_CASH
- FINANCING_CONSISTENCY
- VALUATION_ASSUMPTION_INTEGRITY
- VALUATION_DIAGNOSTIC_INTEGRITY

Local-model qualification applies only where a language model is actually used for assisted research / analytical judgment.

The first Phase C model-dependent module remains:

```text
MOAT_EVIDENCE_AUDIT_ASSISTED_V0_3
```

Additional model-dependent modules require explicit Phase C module-admission artifacts.

### C5 — repeatability

Any candidate that passes a task once must be repeated on unstable or decision-sensitive tasks.

Repeatability requires the same packet, prompt hash, schema hash, model identity, quantization, runtime and generation parameters.

Do not change prompt and model simultaneously when diagnosing variance.

### C6 — blinded / randomized human adjudication

Where multiple candidates are compared, model labels should be hidden from the adjudicator when practical, output order randomized, quality judged against evidence/contract rather than reputation, and deterministic validation must not substitute for human semantic review.

Targeted probes remain calibration evidence and must not be mixed with normal priority-selection scores.

### C7 — local production-candidate decision

A local model may become a production candidate only after all critical semantic regressions PASS, structured-output reliability is acceptable, no critical evidence-role inversion remains, judgment boundary is respected, normal-path usefulness is acceptable, repeatability is acceptable, and machine resource use is operationally viable.

Phase C may identify LOCAL_CANDIDATE_ACCEPTED, LOCAL_CANDIDATE_REJECTED, or LOCAL_CANDIDATE_CONDITIONAL.

Phase C must not create a production routing freeze by implication.

## 5. Candidate evaluation dimensions

Each normal comparison observation must capture engineering validity, human quality, and runtime economics.

Engineering includes schema validity, semantic validator result, malformed-reference errors, finish reason, retry count, runtime errors, context overflow and output truncation.

Human quality includes exact evidence grounding, claim atomicity, claim-target alignment, support/counterevidence polarity, qualification orthogonality, conflict handling, weak-link usefulness, unresolved-point usefulness, judgment-boundary compliance and priority-selection usefulness.

For local models capture external API cost = 0, wall-clock latency, tokens/second where available, peak VRAM where available, peak RAM where available, model file size, quantization and cold-start time where material.

For any later explicitly authorized paid benchmark capture gateway cost, latency, input/output/reasoning tokens.

## 6. Critical-error taxonomy

Critical defects include fabricated evidence IDs, material evidence-role inversion, unsupported financial fact presented as established, failure to preserve explicit UNKNOWN, silent contradiction of a pinned packet, hidden cloud/model fallback, and output accepted under the wrong model identity.

A candidate with recurrent critical errors cannot be admitted merely because its average score is high.

## 7. Initial local candidate policy

The initial candidate registry is maintained separately in:

```text
calibration/vnext/OROTITAN_GATE18_PHASE_C_LOCAL_MODEL_CANDIDATES_V0.1.json
```

The registry is provisional until C0 hardware preflight is observed.

No candidate is admitted solely because benchmark claims are strong.

Hardware fit, exact local runtime behavior and OroTitan-specific regression performance control admission.

## 8. Current candidate ordering principle

Before machine hardware is known:

```text
do not download every candidate
do not run every candidate
do not choose by popularity
```

Instead: inspect local hardware, select the smallest plausible candidate, run C2, run C3, and only move upward in model size if quality fails for a model-capability reason.

## 9. No hidden fallback

All local Phase C execution must fail closed.

Forbidden: automatically calling Vercel AI Gateway when local inference fails; silently switching model or quantization; silently reducing context; silently changing prompt version; silently relaxing schema validation.

A failed local candidate remains a failed local candidate.

## 10. Phase C entry state

```text
Phase A = PASS
Phase B = PASS
Phase C = IN_PROGRESS

local runtime admitted = NO
local hardware preflight = NOT YET OBSERVED
local model candidate admitted = NO
paid model call authorized = NO
model winner = NONE
routing freeze = NONE
production mutation = FALSE
publication authority = FALSE

Gate 18 = IN_PROGRESS_NOT_FROZEN
```

## 11. Immediate next action

The next action is C0 only:

```text
RUN LOCAL HARDWARE PREFLIGHT
```

No inference and no model download should occur before C0 is reviewed.
