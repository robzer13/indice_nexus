# OROTITAN_VNEXT_MODEL_CALIBRATION_V0.1

**Project:** OroTitan Equity Research  
**Gate:** 18  
**Status:** CALIBRATION CANDIDATE — NOT FROZEN  
**Methodology change:** NO  
**Production mutation:** FORBIDDEN  
**Publication authority:** DISABLED  
**Depends on:** Gate 17 Shadow Runner PASS / FROZEN

## 0. Purpose

Gate 18 calibrates physical analytical models empirically.

It does not change analytical methodology and does not pre-select a winning
model.

Core law:

```text
SAME POINT-IN-TIME INPUT
+ SAME MODULE QUESTION
+ SAME GENERATION SCHEMA
+ SAME SOURCE ACCESS
+ SAME OUTPUT CONTRACT
-> COMPARE PHYSICAL MODEL BEHAVIOR
```

The first Gate 18 phase is a five-company pilot.

```text
5 COMPANIES
= CALIBRATION PILOT
!= STATISTICAL VALIDATION
!= ROUTING FREEZE
```

## 1. Current physical candidates

The first controlled OpenAI-family calibration set is:

```text
LUNA  = openai/gpt-5.6-luna
TERRA = openai/gpt-5.6-terra
SOL   = openai/gpt-5.6-sol
ASTRA = openai/gpt-6-astra
```

These are physical candidates only.

They do not alter the abstract routing classes and are not declared permanent
winners.

Initial intended roles:

```text
LUNA  -> T1_STRUCTURED_EXTRACTION
TERRA -> T2_STANDARD_ANALYSIS
SOL   -> T3_PREMIUM_REASONING reference-quality baseline
ASTRA -> T4_FRONTIER_ESCALATION candidate
```

Sol is a reference-quality baseline, not ground truth.

## 2. Pilot corpus

Artifact:

```text
calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json
```

Exactly five functional cases:

```text
RATIONAL AG            SIMPLE_CLEAN_COMPOUNDER
Constellation Software SERIAL_ACQUIRER
STMicroelectronics     CYCLICAL_SEGMENTED
Brookfield Corporation ACCOUNTING_HEAVY
Adyen                   DIFFICULT_MOAT_CONFLICTING_EVIDENCE
```

Every pilot member pins:

```text
SOURCE_RUN_ID
DATA_CUTOFF
Research EVIDENCE_LEDGER artifact/version/hash/repository/commit/path
Research CONFLICT_LEDGER artifact/version/hash/repository/commit/path
```

Only point-in-time Research artifacts are calibration inputs.

Published V2 snapshot conclusions are not model inputs.

## 3. Anti-answer-leakage boundary

Forbidden calibration input:

```text
published V2 conclusion
V2 Investment Score
V2 OQS / OVS result
V2 valuation conclusion
V2 next action
V2 OroTitan terminal result
V2 canonical snapshot payload
```

Allowed comparison/adjudication after generation:

```text
pinned Research evidence
pinned Research conflicts
deterministic calculations
human adjudication
frozen V2 result as historical comparison reference
```

V2 historical output is not ground truth.

## 4. Calibration dimensions

Gate 18 records distributions, not only aggregate scores.

Primary factual dimensions:

```text
factual accuracy
evidence coverage
source/evidence attribution accuracy
unsupported material claims
material fabrication
contradiction detection
```

Primary analytical dimensions:

```text
causal reasoning
counter-evidence handling
weak-link detection
moat mechanism reasoning
runway reasoning
owner-cash reasoning
capital-allocation reasoning
accounting interpretation
valuation-assumption integrity
```

Engineering dimensions:

```text
structured-output success
semantic validation success
repeatability
latency
input tokens
output tokens
reasoning tokens
cached tokens when available
retry count
cost / price receipt when available
```

## 5. No premature quality floor

Gate 18 must not silently convert prior candidate thresholds into frozen
authority.

Historical candidate thresholds may be displayed as hypotheses, but:

```text
NO THRESHOLD FREEZE
NO MODEL PROMOTION
NO ROUTING WINNER
NO COST-QUALITY TRADEOFF RULE
```

until empirical pilot evidence has been inspected and then expanded beyond the
pilot where required.

## 6. Smoke phase

Before company-level calibration, all four candidate models must pass the same
transport/schema smoke.

Smoke artifact:

```text
app/api/vnext/calibration/gate18-smoke/route.ts
```

The smoke is preview-only.

Production behavior:

```text
VERCEL_ENV = production
-> HTTP 403
-> NO MODEL CALL
```

The smoke tests only:

```text
physical model availability
AI Gateway authentication
structured output
schema validation
latency capture
token usage capture
finish reason
provider metadata capture
```

It does not test analytical quality.

## 7. Same-request invariant

For one benchmark task, model-variable fields may differ only where the model
requires a reasoning profile.

The following must remain identical:

```text
company
DATA_CUTOFF
Evidence Packet
module question
system instruction
GenerationSchema
max output contract
tool availability
source availability
```

No candidate may receive additional evidence unavailable to another candidate.

## 8. Repeatability

One successful run is not sufficient to infer stability.

Where outputs are non-deterministic or decision-sensitive, Gate 18 must retain
repeated-run evidence before routing freeze.

The exact repeat count is not frozen in v0.1.

## 9. Ground-truth and adjudication law

```text
MODEL CONSENSUS != GROUND TRUTH
SOL OUTPUT != GROUND TRUTH
ASTRA OUTPUT != GROUND TRUTH
V2 OUTPUT != GROUND TRUTH
```

Ground truth is established from:

```text
pinned evidence
+ deterministic calculations
+ explicit human adjudication where needed
```

## 10. Cost law

A cheaper model may not be promoted solely because it costs less.

A more expensive model may not be promoted solely because it is newer or
larger.

Cost optimization occurs only after quality eligibility is established.

No permanent Gate 18 budget envelope is frozen in v0.1.

## 11. No silent fallback

A requested physical model substitution must be explicit and traceable.

```text
REQUESTED MODEL UNAVAILABLE
-> RECORD FAILURE
-> DO NOT SILENTLY SUBSTITUTE
```

Provider failover for the same physical model may be observed separately, but
must remain visible in provider metadata.

## 12. Current implementation artifacts

```text
MODEL CANDIDATES
= runtime/vnext/model-calibration.ts

FIVE-COMPANY PILOT
= calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json

AI GATEWAY SMOKE
= app/api/vnext/calibration/gate18-smoke/route.ts

DETERMINISTIC ASSURANCE
= tests/vnext-model-calibration.test.ts

RAW CALIBRATION EVIDENCE MODEL
= runtime/vnext/model-calibration-evidence.ts

RAW EVIDENCE TESTS
= tests/vnext-model-calibration-evidence.test.ts

FIRST REAL SMOKE ATTEMPT
= calibration/vnext/OROTITAN_GATE18_SMOKE_ATTEMPT_001.json
```

## 13. Gate 18 progression

Phase A:

```text
4/4 physical model smoke
```

Phase B:

```text
5-company identical-input pilot
quality + engineering receipts
no winner freeze
```

Phase C:

```text
expand paired evidence to diversified corpus / company × module pairs
blind or randomized-label adjudication where material
repeat unstable modules
inspect distributions / critical errors / sensitivity
```

Only after sufficient empirical evidence may a separate routing-policy freeze
be proposed.

## 14. Out of scope

Gate 18 does not:

- change Research or Deep Dive methodology;
- change OQS / OVS / Investment Score;
- change valuation conventions;
- change thresholds, caps or gates;
- authorize publication;
- mutate production;
- make model self-confidence authoritative;
- make the newest model the default by assumption;
- make provider routing opaque;
- freeze Budget Governor amounts from prior candidate documents;
- promote a model before evidence.

## 15. Current gate condition

```text
GATE 18
= IN PROGRESS
= NOT FROZEN
```

Required before Gate 18 freeze:

```text
real model smoke evidence
+
real company-level calibration evidence
+
quality adjudication
+
cost/latency/usage evidence
+
no production mutation
+
deterministic CI
+
separate freeze artifact
```


## 16. First real provider-admission attempt

A real preview invocation was executed after the preview protection boundary was
successfully crossed.

Observed sequence:

```text
PROTECTED PREVIEW
-> SHARE BYPASS ACCEPTED
-> GATE 18 ROUTE REACHED
-> VERCEL AI GATEWAY REACHED
-> HTTP 403 customer_verification_required
```

Vercel AI Gateway rejected the request before a physical model generation was
confirmed because the OROTITAN team does not currently have a valid credit card
on file.

Classification:

```text
EXTERNAL_BILLING_VERIFICATION_REQUIRED
!= MODEL QUALITY FAILURE
!= ANALYTICAL METHODOLOGY FAILURE
!= SHADOW RUNNER FAILURE
```

Therefore:

```text
VALID PHYSICAL MODEL GENERATIONS = 0
VALID QUALITY OBSERVATIONS = 0
MODEL WINNER = NONE
ROUTING CHANGE = NONE
```

A retry without an external billing/provider-access change is not expected to
produce new calibration evidence.

## 17. Raw evidence law

Every future real calibration call must produce a traceable engineering receipt
before it can enter quality adjudication.

The raw receipt model records:

```text
case / company / source run / DATA_CUTOFF
module identity
physical model identity
repetition
prompt identity + version + SHA-256
generation schema identity + version + SHA-256
evidence packet SHA-256
execution / provider request identity
schema + semantic validation state
latency
token usage
retry count
cost + provenance when available
finish reason
response SHA-256
```

Quality adjudication is a separate artifact bound to the exact execution.

It records raw counts such as:

```text
critical / material / minor factual errors
unsupported material claims
required findings covered
material conflicts detected
material counter-evidence handled
critical decision error
supporting evidence references
```

Gate 18 v0.1 deliberately does not convert those observations into:

```text
COMPOSITE MODEL SCORE
RANK
WINNER
TIER PROMOTION
```

Those decisions require later explicit calibration authority.


## 18. Private evidence transport boundary

The implementation repository `robzer13/indice_nexus` is public.

The source artifact repository `robzer13/real-orotitan` is private.

Therefore Gate 18 explicitly forbids:

```text
COPY PRIVATE EVIDENCE_LEDGER CONTENT INTO indice_nexus
COPY PRIVATE CONFLICT_LEDGER CONTENT INTO indice_nexus
COMMIT PRIVATE SOURCE EXCERPTS INTO A PUBLIC FIXTURE
COMMIT GITHUB / PROVIDER / VERCEL ACCESS TOKENS
PASS A V2 FINAL CONCLUSION AS A SUBSTITUTE FOR PRIVATE EVIDENCE
```

The public pilot manifest may contain only non-secret provenance metadata
required to identify and integrity-check the pinned source artifacts:

```text
artifact id
version
SHA-256
private repository name
private path
immutable commit SHA
SOURCE_RUN_ID
DATA_CUTOFF
```

A real company-level calibration run therefore requires an authorized private
evidence transport that:

```text
reads the exact pinned private artifact
verifies its SHA-256 before inference
keeps credentials server-side
does not write the private packet to the public repository
does not expose packet contents in CI logs
does not mutate the source artifact
does not load V2 final conclusions into the model request
```

No such credential may be invented, embedded or inferred by Gate 18.

## 19. Preview-protection transport

The temporary share-link bypass used to diagnose the first real smoke attempt
was a one-time diagnostic path.

Because `indice_nexus` is public, Gate 18 does not retain a workflow that
carries a Vercel share token through public PR metadata.

The retained workflow uses:

```text
GitHub OIDC
-> Vercel Trusted Source
-> protected preview
```

and requires the Vercel project to authorize GitHub Actions as a Trusted
Source before it can invoke protected calibration routes.

This transport decision changes no analytical methodology.
