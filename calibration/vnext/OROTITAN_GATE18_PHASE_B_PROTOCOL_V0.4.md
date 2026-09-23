# OROTITAN_GATE18_PHASE_B_PROTOCOL_V0.4

**Gate:** 18  
**Phase:** B company calibration  
**Status:** EXECUTION CANDIDATE — NOT FROZEN  
**Methodology change:** NO  
**Prompt/schema change:** NO  
**Publication authority:** DISABLED  
**Production mutation:** FORBIDDEN  

## 0. Trigger

The first STMicroelectronics × SOL company-level attempt under protocol v0.3 failed before emitting visible structured output.

Observed:

```text
reasoning = high
maxOutputTokens = 1536
finishReason = length
outputTokens = 1536
reasoningTokens = 1536
generatedTextChars = 0
gatewayCostUsd = 0.043277
```

Vercel AI Gateway documents that GPT-5.6 Sol reasoning tokens count toward `maxOutputTokens`.

Therefore the observed failure is classified as:

```text
TRANSPORT ENVELOPE FAILURE
!= ANALYTICAL QUALITY FAILURE
!= SCHEMA QUALITY FAILURE
```

## 1. Same-request invariant

Gate 18 section 7 requires the following to remain identical for one benchmark task:

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

A SOL-only token-cap increase is therefore forbidden.

## 2. v0.4 correction

Protocol v0.4 changes only the common transport envelope:

```text
common maxOutputTokens: 1536 -> 4096
```

The value is identical for every candidate.

Unchanged:

```text
Evidence Packet
MOAT_INPUTS scope
module question
system prompt
prompt template id/version/hash
GenerationSchema id/version/hash
candidate model identities
candidate reasoning profiles
semantic validation
publication authority
production mutation boundary
```

The 4096 ceiling provides reasoning headroom for high-reasoning candidates while preserving the same bounded visible structured-output schema.

## 3. Comparison consequence

The prior LUNA and TERRA v0.3 outputs remain valid historical calibration observations.

They are not treated as the paired v0.4 comparison set because the max output contract changed.

Before comparing physical-model behavior under v0.4:

```text
LUNA  -> rerun under common 4096 envelope
TERRA -> rerun under common 4096 envelope
SOL   -> rerun only after LUNA/TERRA v0.4 engineering validation
ASTRA -> no paid execution until SOL v0.4 is inspected
```

No automatic retries are authorized.

## 4. Cost discipline

Every paid call remains:

```text
single model only
explicit --model required
explicit --max-case-spend-usd required
pre-call conservative live-price guard
no --allow-multi-model unless separately authorized
```

Dry runs must precede paid calls.

The conservative guard prices the full 4096-token common envelope. This is intentionally stricter than expected usage.

## 5. Gate state

```text
Gate 18 = IN PROGRESS
Phase A = PASS
Phase B = IN PROGRESS
model winner = NONE
routing freeze = NONE
```
