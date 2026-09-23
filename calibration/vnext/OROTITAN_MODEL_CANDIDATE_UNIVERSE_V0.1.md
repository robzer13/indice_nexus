# OROTITAN_MODEL_CANDIDATE_UNIVERSE_V0.1

**Status:** CANDIDATE FOR CALIBRATION — NOT FROZEN  
**Date:** 2026-09-22  
**Methodology change:** NO  
**Purpose:** define the benchmark universe without selecting a winner.

## 0. Principle

Gate 13 validates the provider boundary. It does not select the permanent OroTitan analytical model.

The model-selection problem is deferred to calibration and must compare functional model classes rather than assume a fixed provider chain.

GPT-5.6 Sol High remains the current **reference-quality baseline** for difficult OroTitan reasoning tasks. It is not declared ground truth and is not pre-declared the winner.

## 1. Functional tiers

```text
TIER 0 — DETERMINISTIC
  no LLM

TIER 1A — DOCUMENT PARSING
  specialized OCR / document intelligence

TIER 1B — STRUCTURED EXTRACTION
  low-cost schema-capable models

TIER 2 — STANDARD ANALYSIS
  evidence synthesis / normal business analysis

TIER 3 — PREMIUM REASONING
  decision-sensitive causal / forensic / valuation work

TIER 4 — ADVERSARIAL
  cross-family reviewer selected by benchmark
```

## 2. Mandatory benchmark families

The first broad calibration universe must include, where technically and commercially accessible:

```text
OpenAI
Anthropic
Google
xAI
DeepSeek
Mistral
```

Targeted challengers:

```text
Microsoft AI
Cohere
Moonshot / Kimi
Alibaba / Qwen
Open-weight alternatives
```

Exact physical model/version/provider availability is resolved at benchmark time. No Stage Contract may name a physical model.

## 3. Candidate mapping

### TIER 1A — Document parsing

Candidate classes:

```text
Mistral OCR
Azure Document Intelligence
Azure Content Understanding
other specialized document parsers
```

Architecture under test:

```text
document parser
-> semantic extraction model
-> deterministic validation
```

### TIER 1B — Structured extraction

Candidate families include:

```text
GPT-5.6 Luna
Gemini Flash-class
DeepSeek Flash-class
Mistral Small/Large-class
```

### TIER 2 — Standard analysis

Candidate families include:

```text
GPT-5.6 Terra
Gemini Flash-class
Grok standard reasoning
DeepSeek Pro-class
Mistral Medium-class
Cohere Command-class
Kimi
Qwen challenger
```

### TIER 3 — Premium reasoning

Reference baseline:

```text
GPT-5.6 Sol High
```

Benchmark challengers include:

```text
GPT-6 Astra
Claude Opus / Sonnet premium
Grok reasoning variants
DeepSeek Pro high/max
Gemini high-reasoning configuration
MAI-Thinking shadow
```

### TIER 4 — Adversarial

The reviewer is selected from the cross-family benchmark.

No provider is pre-declared the best adversarial reviewer.

Candidate pairings should include:

```text
Sol -> Anthropic reviewer
Sol -> xAI reviewer
Sol -> Google reviewer
Sol -> DeepSeek reviewer
```

## 4. Calibration law

The benchmark must use identical:

```text
DATA_CUTOFF
Evidence Packet
prompt version
GenerationSchema
module question
source access
```

for all compared models.

Primary dimensions:

```text
factual accuracy
evidence coverage
unsupported material claims
contradiction detection
causal reasoning
counter-evidence discovery
weak-link detection
schema compliance
semantic validation
repeatability
latency
token usage
cost
retry rate
```

A cheaper model may replace the current reference only if it meets an explicit quality floor for the module under test.

## 5. Structured-output architecture

Canonical OroTitan schemas and provider generation schemas are separate artifacts.

```text
CANONICAL OROTITAN SCHEMA
!=
LLM GENERATION SCHEMA
```

Target path:

```text
provider
-> small versioned GenerationSchema
-> constrained decoding when available
-> local JSON/schema validation
-> semantic validator
-> deterministic mapper
-> canonical OroTitan schema
```

Every generation schema must eventually carry:

```text
generation_schema_id
generation_schema_version
generation_schema_sha256
```

## 6. Routing authority

Physical model routing belongs in a versioned RoutingPolicy, not analytical contracts.

Target abstract classes:

```text
EXTRACTION_FAST
STANDARD_ANALYSIS
PREMIUM_REASONING
FRONTIER_ESCALATION
INDEPENDENT_REVIEW
```

No silent fallback is allowed. Every physical-model substitution becomes an auditable routing event.

## 7. Gate 13 boundary

For Gate 13, one real Azure deployment is enough to prove:

```text
AnalyticalModelProvider abstraction
Azure adapter
Responses API
strict structured output
local schema validation
usage capture
quota/rate-limit capture when supplied
traceable cost receipt
authority isolation
```

The initial Azure smoke model does not become the permanent OroTitan model by passing Gate 13.

## 8. Freeze prohibition

This candidate universe must not be frozen before empirical OroTitan calibration.

The permanent routing policy is a later decision and must be evidence-driven.
