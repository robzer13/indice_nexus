# OROTITAN MODEL ROUTING ARCHITECTURE — EXPANDED COMPETITION v0.2

**Project:** OroTitan Equity Research  
**Date:** 2026-09-22  
**Status:** CANDIDATE FOR CALIBRATION — NOT FROZEN  
**Reference-quality baseline:** GPT-5.6 Sol — HIGH  
**Architecture principle:** provider-neutral, deterministic routing authority

## 1. Executive conclusion

OroTitan must not adopt:

- one universal model;
- one universal provider;
- a simple cheap/premium binary;
- an opaque provider-controlled router;
- the current winner of a generic benchmark.

The candidate routing architecture is:

```text
TIER 0 — DETERMINISTIC
        ↓
TIER 1 — DOCUMENT / EXTRACTION
        ↓
TIER 2 — STANDARD ANALYSIS
        ↓
TIER 3 — PREMIUM REASONING
        ↓
TIER 4 — ADVERSARIAL / FRONTIER ESCALATION
        ↓
HUMAN_REQUIRED
```

Physical models behind each tier remain replaceable and must be benchmarked module by module.

The architectural rule is:

> Freeze the routing architecture, not the current model winners.

## 2. Competition universe

Mandatory benchmark families:

```text
OpenAI GPT-5.6 / GPT-6
Anthropic Claude
Google Gemini
xAI / Grok
DeepSeek
Mistral
Alibaba / Qwen
```

Targeted shadow / challenger families:

```text
Microsoft MAI
Cohere
Moonshot / Kimi
GLM
GPT-OSS / Phi / other open-weight candidates
```

A model enters the analytical competition only if it plausibly satisfies a useful combination of:

```text
general analytical capability
financial / quantitative reasoning
long context
structured output
API stability
document understanding
latency
competitive cost
auditability
provider availability
```

Code-only, speech/audio, image-generation, embeddings, rerankers and clearly dominated legacy models are outside the primary analytical competition.

## 3. Functional tier definitions

### TIER 0 — DETERMINISTIC

No LLM.

Includes:

```text
financial arithmetic
ratios
DCF computation
scenario arithmetic
SHA
state machine
Registry
lineage
schema validation
cross-field deterministic controls
scoring formula
publication controls
```

### TIER 1A — DOCUMENT PARSING

Nominal path:

```text
native extraction
→ specialized OCR/document parser only when needed
→ semantic extraction
→ deterministic validation
```

Candidate technologies include Mistral OCR and Azure document-processing services.

### TIER 1B — STRUCTURED EXTRACTION

Calibration incumbent:

```text
GPT-5.6 Luna / reasoning none
```

Mandatory challengers:

```text
Mistral Small-class
Qwen Flash-class
DeepSeek Flash-class
Gemini Flash-class
```

Promotion is based on extraction accuracy and schema reliability, not generic benchmark rank.

### TIER 2 — STANDARD ANALYSIS

Calibration incumbent:

```text
GPT-5.6 Terra / reasoning medium
```

Mandatory challengers:

```text
Gemini Flash-class
Mistral Large-class
Mistral Medium-class
DeepSeek Flash / Pro-class
Qwen Max-class
Claude Sonnet-class
Grok
```

Secondary challengers:

```text
Cohere Command-class
Kimi
```

T2 is expected to have the largest replacement potential.

### TIER 3 — PREMIUM REASONING

Reference-quality baseline:

```text
GPT-5.6 Sol
reasoning = HIGH
```

Mandatory competition:

```text
GPT-6 Astra
Claude Opus-class
Claude frontier-class
Grok frontier reasoning
DeepSeek Pro / Flash high reasoning
Qwen Max-class
Gemini high reasoning
Claude Sonnet high
```

Shadow:

```text
Kimi frontier candidates
MAI-Thinking
```

No model replaces Sol High before module-level non-inferiority is demonstrated.

### TIER 4 — ADVERSARIAL / FRONTIER ESCALATION

Two distinct functions:

```text
T4-A INDEPENDENT_REVIEW
T4-B FRONTIER_TIE_BREAK
```

Independent reviewer candidates should preferably come from a different model family than the primary.

No T4 winner is frozen before calibration.

## 4. Routing engine

Routing is deterministic.

### Step 1 — required capability

The router receives:

```text
TASK_CLASS
MODULE_ID
SCHEMA_RIGIDITY
MODALITY
CONTEXT_VOLUME
REASONING_COMPLEXITY
MODULE_CRITICALITY
EVIDENCE_QUALITY
CONFLICT_STATE
FINANCIAL_MATERIALITY
CONCLUSION_SENSITIVITY
VALUATION_RELIABILITY
WEAK_LINK_STATE
DATA_RESIDENCY_REQUIREMENT
LATENCY_REQUIREMENT
```

### Step 2 — eligibility filter

A physical model is ineligible if it fails any mandatory condition:

```text
benchmark quality floor
schema capability
context capacity
model availability
provider availability
deployment availability
quota
data governance
lifecycle policy
budget hard constraint
```

### Step 3 — constrained optimization

Only eligible models compete.

```text
minimize(
  inference_cost
  + expected_retry_cost
  + schema_repair_cost
  + latency_cost
  + expected_human_review_cost
)

subject to:

QUALITY >= REQUIRED_QUALITY_FLOOR
```

Cost never overrides a hard quality floor.

## 5. Escalation decision tree

```text
REQUEST
│
├── exhaustive deterministic rule?
│       YES → T0
│
├── document parse / mechanical extraction?
│       YES → T1
│
├── standard analysis
│   + adequate evidence
│   + no material conflict
│   + low/moderate sensitivity
│   + no weak-link signal?
│       YES → T2
│
└── HARD_REASONING_TRIGGER?
        ├── material conflict
        ├── weak link
        ├── low execution confidence
        ├── material causal inference
        ├── high financial materiality
        ├── high conclusion sensitivity
        ├── LOW valuation reliability
        ├── weak material evidence
        └── prior semantic failure

        YES → T3

T3 result
│
├── materially resolved → continue
└── unresolved → T4 independent review

T4
│
├── evidence resolves disagreement → reconcile
└── material disagreement persists → HUMAN_REQUIRED
```

Self-reported model confidence is never sufficient by itself to change tier.

## 6. Budget Governor

The Budget Governor is external to the model.

States:

```text
ALLOW
ALLOW_FROM_CRITICAL_RESERVE
DENY_SOFT_ESCALATION
HUMAN_OVERRIDE_REQUIRED
```

Controlled dimensions:

```text
MAX_COST_PER_CALL
MAX_COST_PER_MODULE
MAX_COST_PER_COMPANY
MAX_COST_PER_RUN
MAX_DAILY_COST

INPUT_TOKENS
CACHED_INPUT_TOKENS
CACHE_WRITE_TOKENS
OUTPUT_TOKENS
REASONING_TOKENS

PREMIUM_ESCALATIONS
SECOND_OPINIONS
RETRIES
TOOL_CALLS
```

Calibration envelopes:

```text
ROUTINE               target ~3 USD   hard cap 6 USD
MATERIAL_ANALYSIS     target ~8 USD   hard cap 15 USD
CRITICAL_ANALYSIS     target ~20 USD  hard cap 30 USD
EXCEPTIONAL_REVIEW    human approval
```

Initial reserve guidance:

```text
30–35% of company hard cap reserved for T3/T4
```

Critical rule:

```text
HARD ESCALATION REQUIRED
+
BUDGET INSUFFICIENT
!= downgrade model

=> HUMAN_OVERRIDE_REQUIRED
```

## 7. Adversarial review

Second-pass review is not systematic.

Candidate triggers:

```text
borderline moat
material accounting anomaly
Owner Cash uncertainty
M&A / capital allocation weak link
LOW valuation reliability
high valuation sensitivity
material evidence conflict
fragile investment synthesis
unresolved T3 disagreement
```

The reviewer receives structured evidence, not private chain-of-thought.

Input packet:

```text
Evidence Ledger
Conflict Ledger
Calculation Ledger
Assumption Register
primary structured findings
supporting evidence IDs
contradicting evidence IDs
critical source excerpts
```

Output contract:

```text
finding_id
claim_id
evidence_ids
disagreement_type
materiality
missing_counterevidence
causal_defect
assumption_defect
reopen_recommendation
```

Benchmark comparison must include same-family extra compute versus different-family reviewer.

## 8. Structured-output architecture

Canonical OroTitan schemas and provider generation schemas are different artifacts.

```text
CANONICAL OROTITAN SCHEMA
!=
PROVIDER GENERATION SCHEMA
```

Pipeline:

```text
Canonical Module Contract
        ↓
Provider-neutral GenerationSchema
        ↓
Provider Adapter
        ↓
constrained JSON / JSON mode
        ↓
LOCAL JSON VALIDATOR
        ↓
SEMANTIC VALIDATOR
        ↓
DETERMINISTIC MAPPER
        ↓
CANONICAL OROTITAN SCHEMA
        ↓
POST_STAGE_CERTIFICATION
```

Every call must eventually pin:

```text
generation_schema_id
generation_schema_version
generation_schema_sha256
```

Schema compliance is not analytical correctness.

## 9. Long-context strategy

One-million-token capacity is not a default context format.

Nominal path:

```text
RAW DOCUMENTS
      ↓
parse / extract
      ↓
Evidence Ledger
+
Conflict Ledger
+
Financial Facts
+
Calculation Ledger
+
Assumption Register
+
critical excerpts
      ↓
BLOCK-SPECIFIC CONTEXT PACK
      ↓
T2 / T3 / T4
```

Operating targets:

```text
premium target      <= 100–150k input tokens
warning             ~180–200k
compression/retrieval before provider long-context cost cliffs
full document dump  exception only
```

Long-context price behavior is provider/model/version specific and must be part of the effective routing metadata.

## 10. Provider-neutral architecture

Logical interface:

```text
AnalyticalModelProvider
│
├── AzureProvider
├── OpenAIProvider
├── AnthropicProvider
├── GeminiProvider
├── XAIProvider
├── DeepSeekProvider
├── MistralProvider
├── AlibabaProvider
├── CohereProvider
└── future adapters
```

Analytical contracts request abstract classes only:

```text
STRUCTURED_EXTRACTION
STANDARD_ANALYSIS
PREMIUM_REASONING
INDEPENDENT_REVIEW
FRONTIER_ESCALATION
```

A versioned `RoutingPolicy` maps model class to physical model.

No silent fallback is permitted.

## 11. Request envelope target

```text
task_class
module_id
model_class
reasoning_profile

generation_schema_id
generation_schema_version
generation_schema_sha256

prompt_template_id
prompt_template_version
prompt_template_sha256

evidence_packet_id
evidence_packet_sha256

max_output_tokens
cache_policy
timeout
idempotency_key
budget_envelope
data_residency_requirement
```

## 12. Response envelope target

```text
provider
model_id
model_version
deployment_id
deployment_type
region

request_id
timestamp
latency_ms

input_tokens
cached_input_tokens
cache_write_tokens
reasoning_tokens
output_tokens

tool_calls
tool_tokens

estimated_cost_usd
actual_cost_usd
price_catalog_version

schema_valid
semantic_valid
finish_reason
refusal_state
retry_count

response_sha256
raw_provider_usage
```

## 13. Common error taxonomy

```text
AUTH_ERROR
QUOTA_EXCEEDED
RATE_LIMIT
CONTEXT_LIMIT
TIMEOUT
TRANSIENT_PROVIDER_ERROR
MODEL_UNAVAILABLE
DEPLOYMENT_UNAVAILABLE
SCHEMA_UNSUPPORTED
SCHEMA_VIOLATION
SEMANTIC_VALIDATION_FAILED
SAFETY_REFUSAL
MAX_TOKENS_INCOMPLETE
BILLING_ERROR
DATA_RESIDENCY_MISMATCH
UNKNOWN_PROVIDER_ERROR
```

Every model substitution creates a `MODEL_ROUTING_EVENT`.

Never silently fall back.

## 14. OroTitan benchmark protocol

Phase 1 calibration:

```text
3–5 existing companies
simple / clean compounder
serial acquirer
cyclical / segmented
accounting-heavy
difficult moat / conflicting evidence
```

Every compared model receives exactly the same:

```text
DATA_CUTOFF
Evidence Packet
prompt
GenerationSchema
sources
module objective
```

### T1 matrix

```text
Luna
Mistral Small-class
Qwen Flash-class
DeepSeek Flash-class
Gemini Flash-class
```

### T2 matrix

```text
Terra
Gemini Flash-class
Mistral Large-class
Mistral Medium-class
Qwen Max-class
DeepSeek Flash-class
DeepSeek Pro-class
Claude Sonnet-class
Grok Azure/direct candidate
Cohere Command-class
Kimi
```

### T3 matrix

```text
Sol High             <- reference
Astra high/max
Claude Opus-class
Claude frontier-class
Grok frontier reasoning
Qwen Max-class
DeepSeek high reasoning
Gemini high reasoning
Claude Sonnet high
Kimi frontier shadow
MAI-Thinking shadow
```

### T4 matrix

```text
same-model second pass
vs
cross-family independent reviewer
```

## 15. Benchmark metrics

Factual:

```text
factual accuracy
evidence coverage
unsupported material claims
hallucination
source attribution accuracy
```

Analytical:

```text
contradiction detection
causal reasoning
counter-evidence discovery
weak-link detection
moat mechanism quality
runway quality
capital-allocation reasoning
owner-cash reasoning
accounting interpretation
valuation assumption integrity
```

Engineering:

```text
first-pass schema compliance
semantic validation success
repeatability
latency p50/p95
input tokens
output tokens
reasoning tokens
cached tokens
cost
retry rate
```

## 16. Reference baseline and ground truth

```text
GPT-5.6 SOL HIGH
= REFERENCE QUALITY BASELINE

GPT-5.6 SOL HIGH
!= GROUND TRUTH
```

Ground truth is based on:

```text
evidence
+
human adjudication
+
deterministic calculations
```

## 17. Candidate acceptance thresholds

T1 calibration candidate:

```text
material-field accuracy >= 99%
schema first-pass >= 99.5%
material fabrication = 0
```

T2 candidate:

```text
no material increase in unsupported claims
evidence coverage non-inferior
conflict recall non-inferior
acceptable schema reliability
```

T3 candidate:

```text
NO NEW CRITICAL DECISION ERROR
```

plus defined non-inferiority on:

```text
causal reasoning
counter-evidence
weak-link detection
valuation integrity
```

A lower-cost model loses if it introduces a material investment error.

## 18. Sample size policy

```text
3–5 companies
= CALIBRATION

!= STATISTICAL VALIDATION
```

Before physical-model routing freeze, increase to approximately 15–20 diversified dossiers or enough `company × module` pairs for paired analysis.

Desired methodology:

```text
blind review
randomized model labels
paired comparisons
repeated runs for unstable modules
bootstrap confidence intervals
critical-error veto
```

## 19. Human-required boundary

Always human-authorized:

```text
GO PUBLISH
change frozen methodology
change score weights
change thresholds
change formulas
change gates

freeze new model
promote model to production tier
approve unbenchmarked model substitution

override hard budget cap
resolve materially unresolved cross-model conflict

identity ambiguity issuer/security
data-license exception
data-residency exception

bypass failed PRE_STAGE_PREFLIGHT
bypass failed POST_STAGE_CERTIFICATION

actual investment decision
actual capital allocation
```

## 20. Gate 13 relationship

Gate 13 remains narrow.

Its job is to prove:

```text
AnalyticalModelProvider abstraction
Azure adapter
real Azure deployment
Responses call
provider-constrained structured output
local schema validation
usage capture
quota/rate-limit capture when supplied
traceable cost receipt
authority isolation
```

A physical model used for Gate 13 is only a capability probe.

It does not become the T1/T2/T3 winner.

## 21. Initial physical calibration bootstrap

The initial Azure calibration bootstrap remains:

```text
T1 = GPT-5.6 Luna
T2 = GPT-5.6 Terra
T3 = GPT-5.6 Sol High
```

This is not a superiority declaration.

It is a convenient controlled starting point because the family is visible in the target Azure environment, has strong schema compatibility and supplies the requested Sol High quality reference.

The expanded competition then attempts to displace incumbents on evidence.

## 22. Freeze policy

The following may later be frozen independently of physical winners:

```text
TIER DEFINITIONS
ROUTING INPUTS
ESCALATION RULES
BUDGET GOVERNOR SEMANTICS
MODEL METADATA SCHEMA
ERROR TAXONOMY
BENCHMARK PROTOCOL
NO-SILENT-FALLBACK RULE
QUALITY-FLOOR RULE
```

The following remain versioned and replaceable:

```text
T1_MODEL
T2_MODEL
T3_MODEL
T4_REVIEWER
T4_FRONTIER_MODEL
```

## 23. Candidate status

```text
STATUS = CANDIDATE FOR CALIBRATION — NOT FROZEN
```

This document defines the expanded competition architecture. It does not authorize promotion of any physical model.
